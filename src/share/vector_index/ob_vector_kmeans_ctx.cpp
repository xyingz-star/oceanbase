/**
 * Copyright (c) 2024 OceanBase
 * OceanBase CE is licensed under Mulan PubL v2.
 * You can use this software according to the terms and conditions of the Mulan PubL v2.
 * You may obtain a copy of Mulan PubL v2 at:
 *          http://license.coscl.org.cn/MulanPubL-2.0
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PubL v2 for more details.
 */

 #define USING_LOG_PREFIX SHARE
 #include "ob_vector_kmeans_ctx.h"
 #include "lib/container/ob_array_array.h"
 #include "share/vector_index/ob_plugin_vector_index_service.h"
 #include "share/vector_index/ob_vector_index_util.h"
 #include "share/allocator/ob_tenant_vector_allocator.h"
 #include "storage/ddl/ob_direct_load_struct.h"
 #include "lib/ob_define.h"
 #include "share/ob_errno.h"
 #include "lib/oblog/ob_log_module.h"
 #include "lib/utility/ob_print_utils.h"
 #include "lib/file/file_directory_utils.h"
 #include "lib/lock/ob_mutex.h"
 #include "lib/container/ob_array.h"
 #include <algorithm>
 #include <cstdarg>
 #include <cstdlib>
 #include <cstdio>
 #include <cmath>
 #include <limits>
 #include <numeric>
 #include <utility>
 #include <vector>
 #include <unistd.h>
 #include <pwd.h>
 #include <cstdint>
 #include <cstring>
 #include <strings.h>
 #include <sys/stat.h>
 
 namespace oceanbase {
 using namespace common;
 namespace share {
 
 namespace {
 // K-means schedule after ObKmeansCtx::init(). Edit only this file to switch A/B without touching headers
 // (rebuild stays limited to this translation unit). ObSingleKmeansExecutor::init still may call
 // set_kmeans_train_strategy() after init() to override.
 constexpr ObKmeansTrainStrategy KMEANS_INIT_DEFAULT_TRAIN_STRATEGY = KTS_FULL_BATCH;//KTS_FULL_BATCH;//KTS_NMBKM;
 // NMBKM only: b=n from start, no doubling (full-batch assignment size). Kept in .cpp so toggling does not recompile dependents of the header.
 constexpr bool NMBKM_FORCE_FULL_BATCH = false;//true;
 // Default divisor for initial NMBKM prefix: b0 = max(1, n / div). Runtime overrides (each kmeans entry):
 //   1) OB_NMBKM_MIN_N_SCALE (observer environment)
 //   2) OB_NMBKM_MIN_N_SCALE_FILE (path; first line: one positive integer)
 //   3) /tmp/ob_nmbkm_min_n_scale (same format; for bench: echo 1 > /tmp/ob_nmbkm_min_n_scale before CREATE INDEX)
 //   4) NMBKM_MIN_N_SCALE
 constexpr int64_t NMBKM_MIN_N_SCALE = 16;
 constexpr float NMBKM_RHO = 10.f; // Newling & Fleuret (2016) default doubling threshold
 // Iteration-time starve-based center relocate (prefix batch): largest-bucket blend + random perturbation.
 // When NMBKM_ITER_STARVE_REINIT_ENABLED is false, the whole block below is off; OB_NMBKM_STARVE_ALPHA is ignored.
 // When true: if prefix cluster j has count < floor(b / (k * alpha)), reinit starving center j from the
 // largest prefix bucket j_max: new = w_mu * mu + w_x * x, mu = sum(j_max)/v_max, x = one random full-dataset
 // vector. Default w_mu = v_max/(v_max+1), w_x = 1/(v_max+1) (equivalent to (sum+x)/(v+1)).
 // w_mu hyperparam: NMBKM_STARVE_RELOCATE_BLEND_MU — negative => w_mu=vm/(vm+1); else fixed in [0,1] (clamped).
 // Starve threshold: OB_NMBKM_STARVE_ALPHA (float; <=0 disables when feature enabled). Default alpha=4.
 constexpr bool NMBKM_ITER_STARVE_REINIT_ENABLED = false;
 constexpr float NMBKM_STARVE_ALPHA_DEFAULT = 4.f;
 constexpr float NMBKM_STARVE_RELOCATE_BLEND_MU = 0.7f; // <0: data-dependent vm/(vm+1); >=0: fixed w_mu on mu vs random x
 // After NMBKM full-data stats: fill sparse IVF lists (count < sparse_thr) by splitting donors (PC1 + median / multi-seg).
 constexpr bool NMBKM_POST_PCA_FILL_EMPTY_ENABLED = true;
 constexpr int NMBKM_PCA_POWER_ITERS = 8;
 // sparse_thr = max(1, min(SPARSE_ABS_MAX, batch_sz / (k * SPARSE_DIV))); batch_sz = last NMBKM prefix size (or n).
 constexpr int64_t NMBKM_POST_PCA_SPARSE_DIV = 4;
#define NMBKM_POST_PCA_SPARSE_ABS_MAX 2
 
 bool try_read_nmbkm_div_from_file(const char *path, int64_t &out)
 {
   if (OB_ISNULL(path) || path[0] == '\0') {
     return false;
   }
   FILE *fp = fopen(path, "r");
   if (OB_ISNULL(fp)) {
     return false;
   }
   int64_t v = 0;
   const int nread = fscanf(fp, "%ld", &v);
   (void)fclose(fp);
   if (nread != 1 || v <= 0) {
     return false;
   }
   out = v;
   return true;
 }
 
 int64_t nmbkm_min_n_scale_from_env()
 {
   const char *e = getenv("OB_NMBKM_MIN_N_SCALE");
   if (OB_NOT_NULL(e) && e[0] != '\0') {
     const int64_t v = atoll(e);
     if (v > 0) {
       return v;
     }
   }
   int64_t from_file = 0;
   const char *path = getenv("OB_NMBKM_MIN_N_SCALE_FILE");
   if (OB_NOT_NULL(path) && path[0] != '\0' && try_read_nmbkm_div_from_file(path, from_file)) {
     return from_file;
   }
   static const char DEFAULT_DIV_FILE[] = "/tmp/ob_nmbkm_min_n_scale";
   if (try_read_nmbkm_div_from_file(DEFAULT_DIV_FILE, from_file)) {
     return from_file;
   }
   return NMBKM_MIN_N_SCALE;
 }
 
 float nmbkm_starve_alpha_from_env()
 {
   const char *e = getenv("OB_NMBKM_STARVE_ALPHA");
   if (OB_NOT_NULL(e) && e[0] != '\0') {
     return static_cast<float>(atof(e));
   }
   return NMBKM_STARVE_ALPHA_DEFAULT;
 }
 
 // Weight on largest-bucket mean mu in starve relocate; (1 - w_mu) on random x.
 static float nmbkm_starve_relocate_blend_mu_weight(const int32_t vm)
 {
   if (NMBKM_STARVE_RELOCATE_BLEND_MU < 0.f) {
     const float vf = static_cast<float>(vm);
     return vf / (vf + 1.f);
   }
   const float w = NMBKM_STARVE_RELOCATE_BLEND_MU;
   return std::max(0.f, std::min(1.f, w));
 }
 
 static inline int64_t nmbkm_starve_count_threshold(const int64_t b, const int64_t k, const float alpha)
 {
   if (b <= 0 || k <= 0 || alpha <= 0.f) {
     return INT64_MAX;
   }
   const double thr = static_cast<double>(b) / (static_cast<double>(k) * static_cast<double>(alpha));
   // floor(thr) can be 0 when b << k*alpha; we still use at least 1 so only prefix counts 0 satisfy
   // (cluster_v[j] < 1). Integer counts >=1 are then not treated as starved in that regime.
   return std::max(static_cast<int64_t>(1), static_cast<int64_t>(std::floor(thr)));
 }
 
 const int64_t N_ITER = 100; // for max iterations
 const int64_t KMEANS_LOG_PATH_MAX = 512;
 const int64_t KMEANS_LOG_IDLE_ROTATE_MS = 120000;  // 2 min idle -> new file for next create index
 char g_kmeans_log_dir[KMEANS_LOG_PATH_MAX] = {0};
 char g_kmeans_log_file_path[KMEANS_LOG_PATH_MAX] = {0};
 FILE *g_kmeans_log_file = nullptr;
 int64_t g_last_kmeans_log_ts = 0;
 lib::ObMutex g_kmeans_log_mutex(common::ObLatchIds::OB_KMEANS_CTX_LOCK);
 
 const char *get_kmeans_log_home_dir()
 {
   const char *home = getenv("HOME");
   if (OB_ISNULL(home) || home[0] == '\0') {
     struct passwd *pw = getpwuid(getuid());
     if (OB_NOT_NULL(pw) && OB_NOT_NULL(pw->pw_dir)) {
       home = pw->pw_dir;
     }
   }
   return home;
 }
 
 void kmeans_log_rotate()
 {
   lib::ObMutexGuard guard(g_kmeans_log_mutex);
   if (OB_NOT_NULL(g_kmeans_log_file)) {
     (void)fclose(g_kmeans_log_file);
     g_kmeans_log_file = nullptr;
   }
 }
 
 void ensure_kmeans_log_file_open()
 {
   int64_t now_ms = ObTimeUtility::current_time_ms();
   if (OB_NOT_NULL(g_kmeans_log_file) && (now_ms - g_last_kmeans_log_ts) <= KMEANS_LOG_IDLE_ROTATE_MS) {
     return;
   }
   lib::ObMutexGuard guard(g_kmeans_log_mutex);
   if (OB_NOT_NULL(g_kmeans_log_file) && (now_ms - g_last_kmeans_log_ts) <= KMEANS_LOG_IDLE_ROTATE_MS) {
     return;
   }
   if (OB_NOT_NULL(g_kmeans_log_file)) {
     (void)fclose(g_kmeans_log_file);
     g_kmeans_log_file = nullptr;
   }
   const char *home = get_kmeans_log_home_dir();
   if (OB_ISNULL(home)) {
     return;
   }
   now_ms = ObTimeUtility::current_time_ms();
   (void)snprintf(g_kmeans_log_dir, KMEANS_LOG_PATH_MAX, "%s/log", home);
   (void)snprintf(g_kmeans_log_file_path, KMEANS_LOG_PATH_MAX, "%s/log/kmeans_%ld.log", home, now_ms);
   (void)FileDirectoryUtils::create_full_path(g_kmeans_log_dir);
   g_kmeans_log_file = fopen(g_kmeans_log_file_path, "a");
 }
 
 void kmeans_log(const char *fmt, ...)
   __attribute__((format(printf, 1, 2)));
 
 void kmeans_log(const char *fmt, ...)
 {
   ensure_kmeans_log_file_open();
   if (OB_ISNULL(g_kmeans_log_file)) {
     return;
   }
   lib::ObMutexGuard guard(g_kmeans_log_mutex);
   if (OB_ISNULL(g_kmeans_log_file)) {
     return;
   }
   const int64_t ts_ms = ObTimeUtility::current_time_ms();
   char buf[1024];
   int n = snprintf(buf, sizeof(buf), "[%ld] ", ts_ms);
   if (n <= 0 || n >= (int)sizeof(buf)) {
     return;
   }
   va_list args;
   va_start(args, fmt);
   int n2 = vsnprintf(buf + n, sizeof(buf) - (size_t)n, fmt, args);
   va_end(args);
   if (n2 < 0) {
     return;
   }
   n += n2;
   if (n >= (int)sizeof(buf) - 1) {
     n = (int)sizeof(buf) - 1;
   }
   if (buf[n - 1] != '\n') {
     buf[n] = '\n';
     buf[n + 1] = '\0';
     n++;
   }
   (void)fprintf(g_kmeans_log_file, "%s", buf);
   (void)fflush(g_kmeans_log_file);
   g_last_kmeans_log_ts = ObTimeUtility::current_time_ms();
 }
 
 // Global balance only: cnt_per_center must come from assigning ALL n samples to k lists (e.g. full search_nearest_center).
 // Do not use NMBKM training-time cluster_v (prefix size b<n) for this — that is prefix-only, not global.
 // Reports: mean / var / imbalance / CV / min / Q1 / median / Q3 / max.
 // imbalance_factor == k * sum(c_i^2) / sum(c_i)^2 on those global counts (==1 iff perfectly uniform).
 void log_cluster_assignment_balance_stats(
     const char *algo_label,
     const int64_t n_samples,
     const int64_t k_lists,
     const int32_t *cnt_per_center,
     const char *scope_label)
 {
   if (OB_ISNULL(cnt_per_center) || k_lists <= 0 || OB_ISNULL(algo_label)) {
     return;
   }
   const char *scope =
       (OB_NOT_NULL(scope_label) && scope_label[0] != '\0') ? scope_label : "full_n";
   int64_t sum = 0;
   uint64_t sum_sq = 0;  // sum of c_i^2; uint64 avoids overflow on c*c for large clusters
   int64_t nonempty = 0;
   int64_t min_c = std::numeric_limits<int64_t>::max();
   int64_t max_c = 0;
   for (int64_t i = 0; i < k_lists; ++i) {
     const int64_t c = static_cast<int64_t>(cnt_per_center[i]);
     sum += c;
     sum_sq += static_cast<uint64_t>(c) * static_cast<uint64_t>(c);
     if (c > 0) {
       ++nonempty;
     }
     if (c < min_c) {
       min_c = c;
     }
     if (c > max_c) {
       max_c = c;
     }
   }
   if (min_c == std::numeric_limits<int64_t>::max()) {
     min_c = 0;
   }
   const double kd = static_cast<double>(k_lists);
   const double mean = static_cast<double>(sum) / kd;
   // Population variance of {c_1..c_k}: one pass via sum_sq — same as second-moment minus mean^2.
   double var_pop = static_cast<double>(sum_sq) / kd - mean * mean;
   if (var_pop < 0.0 && var_pop > -1e-12) {
     var_pop = 0.0;
   }
   const double imbalance_factor =
       (sum > 0) ? (kd * static_cast<double>(sum_sq) / (static_cast<double>(sum) * static_cast<double>(sum))) : 0.0;
   const double std_pop = std::sqrt(std::max(0.0, var_pop));
   const double cv = (mean > 1e-30) ? (std_pop / mean) : 0.0;
   const double max_over_mean = (mean > 1e-30) ? (static_cast<double>(max_c) / mean) : 0.0;
 
   std::vector<int64_t> sorted(static_cast<size_t>(k_lists));
   for (int64_t i = 0; i < k_lists; ++i) {
     sorted[static_cast<size_t>(i)] = static_cast<int64_t>(cnt_per_center[i]);
   }
   std::sort(sorted.begin(), sorted.end());
   const size_t ik = static_cast<size_t>(k_lists > 1 ? k_lists - 1 : 0);
   const size_t i25 = static_cast<size_t>(ik * 25 / 100);
   const size_t i50 = static_cast<size_t>(ik * 50 / 100);
   const size_t i75 = static_cast<size_t>(ik * 75 / 100);
   const int64_t q1 = sorted[i25];
   const int64_t med = sorted[i50];
   const int64_t q3 = sorted[i75];
 
   SHARE_LOG(INFO,
       "cluster_size_stats_global (scope in kmeans_log; full-n assignment to k lists; global uniformity)",
       KCSTRING(algo_label),
       KCSTRING(scope),
       K(n_samples),
       K(k_lists),
       K(sum),
       K(nonempty),
       K(mean),
       K(var_pop),
       K(imbalance_factor),
       K(cv),
       K(max_over_mean),
       K(min_c),
       K(q1),
       K(med),
       K(q3),
       K(max_c));
   kmeans_log(
       "cluster_size_stats_global scope=%s algo=%s n=%ld k=%ld sum=%ld nonempty_lists=%ld mean=%.6f var_pop=%.6f "
       "imbalance_factor=%.6f cv=%.6f max/mean=%.6f min=%ld Q1=%ld median=%ld Q3=%ld max=%ld",
       scope,
       algo_label,
       n_samples,
       k_lists,
       sum,
       nonempty,
       mean,
       var_pop,
       imbalance_factor,
       cv,
       max_over_mean,
       min_c,
       q1,
       med,
       q3,
       max_c);
 }
 }  // namespace

 namespace {
 // Default dev layout: $HOME/test/flash-kmeans (Python venv) + $HOME/test/oceanbase/tools/ob_external_kmeans_worker.py
 // Disable with OB_USE_TEST_FLASH_KMEANS=0 or false. Override command with OB_EXTERNAL_KMEANS_CMD.
 char g_ob_default_ext_kmeans_cmd[2048];
 bool g_ob_default_ext_kmeans_cmd_built = false;
 char g_ob_default_ext_kmeans_pq_batch_cmd[2048];
 bool g_ob_default_ext_kmeans_pq_batch_cmd_built = false;

 const char *ob_get_external_kmeans_cmd_or_default()
 {
   const char *manual = ::getenv("OB_EXTERNAL_KMEANS_CMD");
   if (OB_NOT_NULL(manual) && manual[0] != '\0') {
     return manual;
   }
   if (!g_ob_default_ext_kmeans_cmd_built) {
     g_ob_default_ext_kmeans_cmd_built = true;
     g_ob_default_ext_kmeans_cmd[0] = '\0';
     const char *home = ::getenv("HOME");
     if (OB_NOT_NULL(home) && home[0] != '\0') {
       char py[512];
       char wk[512];
       const int nw = snprintf(wk, sizeof(wk), "%s/test/oceanbase/tools/ob_external_kmeans_worker.py", home);
       if (nw > 0 && nw < static_cast<int>(sizeof(wk)) && 0 == ::access(wk, R_OK)) {
         int npy = snprintf(py, sizeof(py), "%s/test/flash-kmeans/.venv311/bin/python3", home);
         if (npy <= 0 || npy >= static_cast<int>(sizeof(py)) || 0 != ::access(py, X_OK)) {
           (void)snprintf(py, sizeof(py), "%s/test/flash-kmeans/.venv/bin/python3", home);
         }
         if (0 == ::access(py, X_OK)) {
           const int nc = snprintf(g_ob_default_ext_kmeans_cmd,
               sizeof(g_ob_default_ext_kmeans_cmd),
               "%s %s",
               py,
               wk);
           if (nc <= 0 || nc >= static_cast<int>(sizeof(g_ob_default_ext_kmeans_cmd))) {
             g_ob_default_ext_kmeans_cmd[0] = '\0';
           }
         }
       }
     }
   }
   return (g_ob_default_ext_kmeans_cmd[0] != '\0') ? g_ob_default_ext_kmeans_cmd : nullptr;
 }

 const char *ob_get_external_kmeans_pq_batch_cmd_or_default()
 {
   const char *manual = ::getenv("OB_EXTERNAL_KMEANS_PQ_BATCH_CMD");
   if (OB_NOT_NULL(manual) && manual[0] != '\0') {
     return manual;
   }
   if (!g_ob_default_ext_kmeans_pq_batch_cmd_built) {
     g_ob_default_ext_kmeans_pq_batch_cmd_built = true;
     g_ob_default_ext_kmeans_pq_batch_cmd[0] = '\0';
     const char *home = ::getenv("HOME");
     if (OB_NOT_NULL(home) && home[0] != '\0') {
       char py[512];
       char wk[512];
       const int nw = snprintf(wk,
           sizeof(wk),
           "%s/test/oceanbase/tools/ob_external_kmeans_pq_batch_worker.py",
           home);
       if (nw > 0 && nw < static_cast<int>(sizeof(wk)) && 0 == ::access(wk, R_OK)) {
         int npy = snprintf(py, sizeof(py), "%s/test/flash-kmeans/.venv311/bin/python3", home);
         if (npy <= 0 || npy >= static_cast<int>(sizeof(py)) || 0 != ::access(py, X_OK)) {
           (void)snprintf(py, sizeof(py), "%s/test/flash-kmeans/.venv/bin/python3", home);
         }
         if (0 == ::access(py, X_OK)) {
           const int nc = snprintf(g_ob_default_ext_kmeans_pq_batch_cmd,
               sizeof(g_ob_default_ext_kmeans_pq_batch_cmd),
               "%s %s",
               py,
               wk);
           if (nc <= 0 || nc >= static_cast<int>(sizeof(g_ob_default_ext_kmeans_pq_batch_cmd))) {
             g_ob_default_ext_kmeans_pq_batch_cmd[0] = '\0';
           }
         }
       }
     }
   }
   return (g_ob_default_ext_kmeans_pq_batch_cmd[0] != '\0') ? g_ob_default_ext_kmeans_pq_batch_cmd : nullptr;
 }

 static bool ob_external_pq_batch_enabled_by_env()
 {
   const char *v = ::getenv("OB_EXTERNAL_KMEANS_PQ_BATCH");
   if (OB_NOT_NULL(v) && v[0] != '\0') {
     if ((0 == strcmp(v, "0")) || (0 == strcasecmp(v, "false")) || (0 == strcasecmp(v, "off")) ||
         (0 == strcasecmp(v, "no"))) {
       return false;
     }
   }
   return OB_NOT_NULL(ob_get_external_kmeans_pq_batch_cmd_or_default());
 }

 static int ob_external_pq_micro_batch_from_env()
 {
   const char *e = ::getenv("OB_EXTERNAL_KMEANS_PQ_MICRO_BATCH");
   if (OB_NOT_NULL(e) && e[0] != '\0') {
     const int v = atoi(e);
     if (v >= 1 && v <= 256) {
       return v;
     }
   }
   return 16;
 }

 static bool ob_external_kmeans_stage_disabled_by_env(const char *env_name)
 {
   const char *v = ::getenv(env_name);
   return OB_NOT_NULL(v) && v[0] != '\0'
       && ((0 == strcmp(v, "0")) || (0 == strcasecmp(v, "false")) || (0 == strcasecmp(v, "off"))
           || (0 == strcasecmp(v, "no")));
 }

 // is_pq_stage: false = IVF coarse (ObIvfFlatBuildHelper), true = PQ codebook (ObIvfPqBuildHelper).
 // OB_USE_TEST_FLASH_KMEANS=0 -> both stages Elkan.
 // OB_EXTERNAL_KMEANS_IVF=0 -> coarse stays Elkan; PQ still external if worker present (unless OB_EXTERNAL_KMEANS_PQ=0).
 // OB_EXTERNAL_KMEANS_PQ=0 -> PQ stays Elkan; coarse still external if worker present (unless OB_EXTERNAL_KMEANS_IVF=0).
 ObKmeansAlgoType ob_resolve_kmeans_algo_type_from_env(const bool is_pq_stage)
 {
   const char *opt_out = ::getenv("OB_USE_TEST_FLASH_KMEANS");
   if (OB_NOT_NULL(opt_out) && opt_out[0] != '\0') {
     if ((0 == strcmp(opt_out, "0")) || (0 == strcasecmp(opt_out, "false")) || (0 == strcasecmp(opt_out, "off"))) {
       return ObKmeansAlgoType::KAT_ELKAN;
     }
   }
   if (is_pq_stage && ob_external_kmeans_stage_disabled_by_env("OB_EXTERNAL_KMEANS_PQ")) {
     kmeans_log("external_kmeans: OB_EXTERNAL_KMEANS_PQ disabled -> PQ codebook uses Elkan (CPU)");
     return ObKmeansAlgoType::KAT_ELKAN;
   }
   if (!is_pq_stage && ob_external_kmeans_stage_disabled_by_env("OB_EXTERNAL_KMEANS_IVF")) {
     kmeans_log("external_kmeans: OB_EXTERNAL_KMEANS_IVF disabled -> IVF coarse uses Elkan (CPU)");
     return ObKmeansAlgoType::KAT_ELKAN;
   }
   if (OB_NOT_NULL(ob_get_external_kmeans_cmd_or_default())) {
     return ObKmeansAlgoType::KAT_EXTERNAL_GPU;
   }
   return ObKmeansAlgoType::KAT_ELKAN;
 }

 constexpr uint32_t OB_EXT_KMEANS_MAGIC = 0x4D4B424FU;  // 'OBKM' LE
 constexpr uint32_t OB_EXT_KMEANS_VERSION = 1U;
 constexpr uint32_t OB_EXT_KMEANS_FLAG_HAS_INIT_CENTERS = 1U << 0;
 // Worker-side k-means++ init flag; default ON (set OB_EXTERNAL_GPU_KMEANSPP=0|false|off|no to use OB CPU k-means++).
 // Must match tools/ob_external_kmeans_worker.py
 constexpr uint32_t OB_EXT_KMEANS_FLAG_WORKER_GPU_KMEANSPP = 1U << 1;

 // PQ batch worker (ob_external_kmeans_pq_batch_worker.py): one N×full_dim transfer, micro-batch GPU Lloyd.
 constexpr uint32_t OB_EXT_KMEANS_PQ_BATCH_MAGIC = 0x424B424FU;  // 'OBKB' LE
 constexpr uint32_t OB_EXT_KMEANS_PQ_BATCH_VERSION = 1U;

#pragma pack(push, 1)
 struct ObExtKmeansPqBatchInHeader
 {
   uint32_t magic_;
   uint32_t version_;
   int64_t n_samples_;
   int64_t full_dim_;
   int64_t m_;
   int64_t sub_dim_;
   int64_t k_;
   int32_t max_iters_;
   int32_t micro_batch_;
   uint32_t flags_;
 };
 struct ObExtKmeansPqBatchOutHeader
 {
   uint32_t magic_;
   uint32_t version_;
   int64_t m_;
   int64_t k_;
   int64_t sub_dim_;
 };
#pragma pack(pop)

 int write_ob_external_kmeans_pq_batch_input(
     const char *path,
     const ObIArray<float *> &full_vectors,
     const int64_t full_dim,
     const int64_t m,
     const int64_t sub_dim,
     const int64_t k,
     const int max_iters,
     const int micro_batch,
     const bool worker_gpu_kmeanspp)
 {
   int ret = OB_SUCCESS;
   FILE *fp = fopen(path, "wb");
   if (OB_ISNULL(fp)) {
     ret = OB_IO_ERROR;
     SHARE_LOG(WARN, "fopen pq batch input failed", K(ret), KP(path));
   } else {
     ObExtKmeansPqBatchInHeader hdr;
     hdr.magic_ = OB_EXT_KMEANS_PQ_BATCH_MAGIC;
     hdr.version_ = OB_EXT_KMEANS_PQ_BATCH_VERSION;
     hdr.n_samples_ = full_vectors.count();
     hdr.full_dim_ = full_dim;
     hdr.m_ = m;
     hdr.sub_dim_ = sub_dim;
     hdr.k_ = k;
     hdr.max_iters_ = max_iters;
     hdr.micro_batch_ = micro_batch;
     hdr.flags_ = worker_gpu_kmeanspp ? OB_EXT_KMEANS_FLAG_WORKER_GPU_KMEANSPP : 0U;
     if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1) {
       ret = OB_IO_ERROR;
       SHARE_LOG(WARN, "fwrite pq batch header failed", K(ret));
     }
     for (int64_t i = 0; OB_SUCC(ret) && i < full_vectors.count(); ++i) {
       if (fwrite(full_vectors.at(i), sizeof(float) * static_cast<size_t>(full_dim), 1, fp) != 1) {
         ret = OB_IO_ERROR;
         SHARE_LOG(WARN, "fwrite pq batch sample failed", K(ret), K(i));
       }
     }
     if (0 != fclose(fp)) {
       ret = OB_SUCC(ret) ? OB_IO_ERROR : ret;
       SHARE_LOG(WARN, "fclose pq batch input failed", K(ret));
     }
   }
   return ret;
 }

#pragma pack(push, 1)
 struct ObExtKmeansInHeader
 {
   uint32_t magic_;
   uint32_t version_;
   int64_t n_samples_;
   int64_t dim_;
   int64_t k_;
   int32_t max_iters_;
   int32_t dist_algo_;
   uint32_t flags_;
 };
 struct ObExtKmeansOutHeader
 {
   uint32_t magic_;
   uint32_t version_;
   int64_t k_;
   int64_t dim_;
 };
#pragma pack(pop)

 int ob_external_kmeans_max_iters_from_env()
 {
   const char *e = ::getenv("OB_EXTERNAL_KMEANS_MAX_ITERS");
   if (OB_NOT_NULL(e) && e[0] != '\0') {
     const int v = atoi(e);
     if (v > 0 && v <= 10000) {
       return v;
     }
   }
   return 100;
 }

 bool ob_external_gpu_kmeanspp_enabled()
 {
   const char *e = ::getenv("OB_EXTERNAL_GPU_KMEANSPP");
   if (OB_NOT_NULL(e) && e[0] != '\0') {
     if ((0 == strcmp(e, "0")) || (0 == strcasecmp(e, "false")) || (0 == strcasecmp(e, "off")) ||
         (0 == strcasecmp(e, "no"))) {
       return false;
     }
   }
   return true;
 }

 int write_ob_external_kmeans_input(
     const char *path,
     const ObIArray<float *> &input_vectors,
     const ObCentersBuffer<float> &init_centers,
     const int64_t k,
     const int64_t dim,
     const int32_t dist_algo,
     const int max_iters,
     const bool write_init,
     const bool worker_gpu_kmeanspp)
 {
   int ret = OB_SUCCESS;
   FILE *fp = fopen(path, "wb");
   if (OB_ISNULL(fp)) {
     ret = OB_IO_ERROR;
     SHARE_LOG(WARN, "fopen input failed", K(ret), KP(path));
   } else {
     ObExtKmeansInHeader hdr;
     hdr.magic_ = OB_EXT_KMEANS_MAGIC;
     hdr.version_ = OB_EXT_KMEANS_VERSION;
     hdr.n_samples_ = input_vectors.count();
     hdr.dim_ = dim;
     hdr.k_ = k;
     hdr.max_iters_ = max_iters;
     // Must match tools/ob_external_kmeans_worker.py / ObVectorIndexDistAlgorithm (VIDA_L2=0, VIDA_IP=1, VIDA_COS=2).
     hdr.dist_algo_ = dist_algo;
     if (worker_gpu_kmeanspp) {
       hdr.flags_ = OB_EXT_KMEANS_FLAG_WORKER_GPU_KMEANSPP;
     } else {
       hdr.flags_ = write_init ? OB_EXT_KMEANS_FLAG_HAS_INIT_CENTERS : 0U;
     }
     if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1) {
       ret = OB_IO_ERROR;
       SHARE_LOG(WARN, "fwrite header failed", K(ret));
     }
     for (int64_t i = 0; OB_SUCC(ret) && i < input_vectors.count(); ++i) {
       if (fwrite(input_vectors.at(i), sizeof(float) * static_cast<size_t>(dim), 1, fp) != 1) {
         ret = OB_IO_ERROR;
         SHARE_LOG(WARN, "fwrite sample failed", K(ret), K(i));
       }
     }
     if (OB_SUCC(ret) && write_init && !worker_gpu_kmeanspp) {
       for (int64_t c = 0; OB_SUCC(ret) && c < k; ++c) {
         if (fwrite(init_centers.at(c), sizeof(float) * static_cast<size_t>(dim), 1, fp) != 1) {
           ret = OB_IO_ERROR;
           SHARE_LOG(WARN, "fwrite init center failed", K(ret), K(c));
         }
       }
     }
     if (0 != fclose(fp)) {
       ret = OB_SUCC(ret) ? OB_IO_ERROR : ret;
       SHARE_LOG(WARN, "fclose input failed", K(ret));
     }
   }
   return ret;
 }

 void ob_external_kmeans_log_config(
     const bool gpu_kpp, const bool write_init, const int max_iters, const char *cmd)
 {
   const uint32_t bin_flags = gpu_kpp ? OB_EXT_KMEANS_FLAG_WORKER_GPU_KMEANSPP
                                      : (write_init ? OB_EXT_KMEANS_FLAG_HAS_INIT_CENTERS : 0U);
   const char *kpp_env = ::getenv("OB_EXTERNAL_GPU_KMEANSPP");
   const char *manual_cmd = ::getenv("OB_EXTERNAL_KMEANS_CMD");
   const char *opt_flash = ::getenv("OB_USE_TEST_FLASH_KMEANS");
   const char *flash_env = ::getenv("USE_FLASH_KMEANS");
   const char *cuda_vis = ::getenv("CUDA_VISIBLE_DEVICES");
   const char *kpp_seed = ::getenv("OB_KMEANSPP_SEED");
   char cmd_short[400];
   cmd_short[0] = '\0';
   if (OB_NOT_NULL(cmd) && cmd[0] != '\0') {
     const size_t len = strlen(cmd);
     const size_t max_show = 320;
     if (len <= max_show) {
       (void)snprintf(cmd_short, sizeof(cmd_short), "%s", cmd);
     } else {
       (void)snprintf(cmd_short, sizeof(cmd_short), "%.*s...", static_cast<int>(max_show), cmd);
     }
   }
   kmeans_log(
       "external_kmeans_config gpu_kpp=%d write_init=%d bin_flags=0x%x max_iters=%d "
       "OB_EXTERNAL_GPU_KMEANSPP=%s OB_EXTERNAL_KMEANS_CMD=%s OB_USE_TEST_FLASH_KMEANS=%s USE_FLASH_KMEANS=%s "
       "CUDA_VISIBLE_DEVICES=%s OB_KMEANSPP_SEED=%s worker_cmd=%s",
       gpu_kpp ? 1 : 0,
       write_init ? 1 : 0,
       bin_flags,
       max_iters,
       OB_NOT_NULL(kpp_env) && kpp_env[0] != '\0' ? kpp_env : "(unset_default_worker_kpp)",
       OB_NOT_NULL(manual_cmd) && manual_cmd[0] != '\0' ? manual_cmd : "(unset)",
       OB_NOT_NULL(opt_flash) && opt_flash[0] != '\0' ? opt_flash : "(unset)",
       OB_NOT_NULL(flash_env) && flash_env[0] != '\0' ? flash_env : "(unset)",
       OB_NOT_NULL(cuda_vis) && cuda_vis[0] != '\0' ? cuda_vis : "(unset)",
       OB_NOT_NULL(kpp_seed) && kpp_seed[0] != '\0' ? kpp_seed : "(unset)",
       cmd_short[0] != '\0' ? cmd_short : "(null)");
   SHARE_LOG(INFO,
       "external kmeans config",
       K(gpu_kpp),
       K(write_init),
       K(bin_flags),
       K(max_iters),
       KP(cmd));
 }

 }  // namespace
 // ------------------ ObKmeansCtx implement ------------------
 void ObKmeansCtx::destroy()
 {
   for (int i = 0; i < sample_vectors_.count(); ++i) {
     ivf_build_mem_ctx_.Deallocate(sample_vectors_[i]);
   }
   sample_vectors_.reset();
 }
 
 int ObKmeansCtx::init(
     const int64_t tenant_id,
     const int64_t lists,
     const int64_t samples_per_nlist,
     const int64_t dim,
     ObVectorIndexDistAlgorithm dist_algo,
     ObVectorNormalizeInfo *norm_info,
     int64_t pq_m,
     bool is_pq_stage)
 {
   int ret = OB_SUCCESS;
   if (OB_INVALID_ID == tenant_id || 0 >= lists || 0 >= samples_per_nlist || 0 >= dim || VIDA_MAX <= dist_algo
     || dim % pq_m != 0) {
     ret = OB_INVALID_ARGUMENT;
     SHARE_LOG(WARN, "invalid argument", K(ret), K(lists), K(tenant_id), K(samples_per_nlist), K(dim), K(dist_algo));
   } else if (INT64_MAX / samples_per_nlist < lists) {
     ret = OB_INVALID_ARGUMENT;
     LOG_WARN("ivf vector index param nlist_value * sample_per_nlist_value should less than int64_max", K(ret),
              K(lists), K(samples_per_nlist));
     LOG_USER_ERROR(OB_INVALID_ARGUMENT,
                    "ivf vector index param nlist_value * sample_per_nlist_value should less than int64_max");
   } else {
     sample_vectors_.set_attr(ObMemAttr(tenant_id, "KmeansSample"));
     tenant_id_ = tenant_id;
     sample_dim_ = dim;
     dim_ = sample_dim_ / pq_m;
     lists_ = lists;
     max_sample_count_ = lists_ * samples_per_nlist;
     dist_algo_ = dist_algo;
     norm_info_ = norm_info;
     is_pq_stage_ = is_pq_stage; // set is_pq_stage from parameter
     is_inited_ = true;
     train_strategy_ = KMEANS_INIT_DEFAULT_TRAIN_STRATEGY;
   }
   return ret;
 }
 
 int ObKmeansCtx::try_normalize(int64_t dim, float *data, float *norm_vector) const
 {
   int ret = OB_SUCCESS;
   if (OB_NOT_NULL(norm_info_)) { // cos&ip need norm center vec
     if (OB_FAIL(norm_info_->normalize_func_(dim, data, norm_vector, nullptr))) {
       LOG_WARN("failed to normalize vector", K(ret));
     }
   }
   return ret;
 }
 
 int ObKmeansCtx::try_normalize_samples() const
 {
   int ret = OB_SUCCESS;
   if (OB_NOT_NULL(norm_info_) && VIDA_COS == dist_algo_) {  // cos need norm before kmeans
     for (int i = 0; OB_SUCC(ret) && i < sample_vectors_.count(); ++i) {
       if (OB_FAIL(norm_info_->normalize_func_(sample_dim_, sample_vectors_[i], sample_vectors_[i], nullptr))) {
         LOG_WARN("failed to normalize vector", K(ret), K(i), K(sample_dim_));
       }
     }
   }
   return ret;
 }
 
 int ObKmeansCtx::append_sample_vector(float* vector)
 {
   int ret = OB_SUCCESS;
   if (IS_NOT_INIT) {
     ret = OB_NOT_INIT;
     SHARE_LOG(WARN, "kmeans ctx is not inited", K(ret));
   } else if (OB_ISNULL(vector)) {
     ret = OB_INVALID_ARGUMENT;
     SHARE_LOG(WARN, "invalid null vector", K(ret));
   } else {
     // reservoir sampling
     lib::ObMutexGuard guard(lock_);
     if (max_sample_count_ > sample_vectors_.count()) {
       float* save_vector = nullptr;
       if (OB_ISNULL(save_vector = static_cast<float*>(ivf_build_mem_ctx_.Allocate(sizeof(float) * sample_dim_)))) {
         ret = OB_ALLOCATE_MEMORY_FAILED;
         SHARE_LOG(WARN, "failed to alloc vector", K(ret), K(ivf_build_mem_ctx_.get_all_vsag_use_mem_byte()));
       } else {
         MEMCPY(save_vector, vector, sizeof(float) * sample_dim_);
         if (OB_FAIL(sample_vectors_.push_back(save_vector))) {
           SHARE_LOG(WARN, "failed to push back array", K(ret));
         }
       }
     } else {
       int64_t random = 0;
       random = ObRandom::rand(0, total_scan_count_ - 1);
       if (random < sample_vectors_.count()) {
         float* switch_vector = sample_vectors_.at(random);
         MEMCPY(switch_vector, vector, sizeof(float) * sample_dim_);
       }
     }
   }
   return ret;
 }
 
 // ------------------ ObKmeansAlgo implement ------------------
 void ObKmeansAlgo::destroy()
 {
   if (OB_NOT_NULL(weight_)) {
     ivf_build_mem_ctx_.Deallocate(weight_);
     weight_ = nullptr;
   }
   if (OB_NOT_NULL(distance_tasks_)) {
     ivf_build_mem_ctx_.Deallocate(distance_tasks_);
     distance_tasks_ = nullptr;
   }
   if (OB_NOT_NULL(assign_tasks_)) {
     ivf_build_mem_ctx_.Deallocate(assign_tasks_);
     assign_tasks_ = nullptr;
   }
   task_handler_ = nullptr;
   kmeans_monitor_ = nullptr;
 }
 
 bool ObKmeansAlgo::check_stop()
 {
   int ret = OB_SUCCESS;
   if (OB_FAIL(share::dag_yield())) {
     LOG_WARN("dag yield failed", K(ret)); // exit for dag task as soon as possible after canceled.
   } else if (OB_FAIL(THIS_WORKER.check_status())) {
     LOG_WARN("check status failed", K(ret));
   } else if (ATOMIC_LOAD(&force_stop_) == true) {
     ret = OB_CANCELED;
     LOG_WARN("kmeans ctx is fore stop", K(ret), K(*this));
   }
   return OB_SUCCESS != ret;
 }
 
 int ObKmeansAlgo::init(ObKmeansCtx &kmeans_ctx, bool enable_parallel /* = false */)
 {
   int ret = OB_SUCCESS;
   int64_t tenant_id = kmeans_ctx.tenant_id_;
   int64_t lists = kmeans_ctx.lists_;
   int64_t dim = kmeans_ctx.dim_;
   if (OB_INVALID_ID == tenant_id || 0 >= lists || 0 >= dim) {
     ret = OB_INVALID_ARGUMENT;
     SHARE_LOG(WARN, "invalid argument", K(ret), K(lists), K(tenant_id), K(dim));
   } else {
     kmeans_ctx_ = &kmeans_ctx;
     enable_parallel_ = enable_parallel;
     // decide whether to enable HGraph acceleration based on conditions
     // conditions: 1. nlist >= threshold (5000)  2. is IVF clustering stage (not PQ quantization stage)
     const int64_t nlist = kmeans_ctx.lists_;
     const int64_t hgraph_threshold = ObVecIdxExtraInfo::IVF_BUILD_HGRAPH_THRESHOLD;
     // check if is IVF clustering stage (not PQ quantization stage)
     bool is_ivf_stage = !kmeans_ctx.is_pq_stage_ && (kmeans_ctx.dim_ == kmeans_ctx.sample_dim_);
     enable_hgraph_ = is_ivf_stage && (nlist >= hgraph_threshold);
     LOG_DEBUG("HGraph acceleration decision", K(enable_hgraph_), K(is_ivf_stage), K(kmeans_ctx.is_pq_stage_), K(nlist),
              K(hgraph_threshold), K(kmeans_ctx.dim_), K(kmeans_ctx.sample_dim_));
 
     is_inited_ = true;
   }
   return ret;
 }
 
 int ObKmeansAlgo::build(const ObIArray<float*> &input_vectors)
 {
   int ret = OB_SUCCESS;
   if (IS_NOT_INIT) {
     ret = OB_NOT_INIT;
     SHARE_LOG(WARN, "kmeans ctx is not inited", K(ret));
   } else {
     const int64_t kmeans_start_time = ObTimeUtility::current_time_ms();
     SHARE_LOG(INFO, "kmeans build start", K(kmeans_ctx_->lists_), K(kmeans_ctx_->dim_), K(input_vectors.count()));
     kmeans_log("kmeans build start lists=%ld dim=%ld count=%ld", kmeans_ctx_->lists_, kmeans_ctx_->dim_, input_vectors.count());
     ObKMeansStatus last_status = status_;
     int64_t status_start_time = ObTimeUtility::current_time_ms();
     while (OB_SUCC(ret) && !is_finish()) {
       if (OB_FAIL(inner_build(input_vectors))) {
         SHARE_LOG(WARN, "failed to do kmeans", K(ret));
       } else if (check_stop()) {
         ret = OB_CANCELED;
         SHARE_LOG(INFO, "kmeans ctx is fore stop", K(ret), K(*this));
       } else if (last_status != status_) {
         SHARE_LOG(INFO, "status change", K(last_status), K(status_), K(ObTimeUtility::current_time_ms() - status_start_time));
         last_status = status_;
         status_start_time = ObTimeUtility::current_time_ms();
       }
     }
     const int64_t kmeans_cost_ms = ObTimeUtility::current_time_ms() - kmeans_start_time;
     SHARE_LOG(INFO, "kmeans build finished", K(ret), K(kmeans_cost_ms));
     kmeans_log("kmeans build finished ret=%d kmeans_cost_ms=%ld", ret, kmeans_cost_ms);
   }
   return ret;
 }

 int ObKmeansAlgo::load_finished_centers(const float *rows, const int64_t k, const int64_t dim)
 {
   int ret = OB_SUCCESS;
   if (IS_NOT_INIT) {
     ret = OB_NOT_INIT;
     SHARE_LOG(WARN, "kmeans algo is not inited", K(ret));
   } else if (OB_ISNULL(rows) || k <= 0 || dim <= 0) {
     ret = OB_INVALID_ARGUMENT;
     SHARE_LOG(WARN, "invalid load_finished_centers args", K(ret), KP(rows), K(k), K(dim));
   } else if (OB_FAIL(centers_[0].init(dim, k, ivf_build_mem_ctx_))) {
     SHARE_LOG(WARN, "failed to init centers buffer", K(ret));
   } else if (OB_FAIL(centers_[1].init(dim, k, ivf_build_mem_ctx_))) {
     SHARE_LOG(WARN, "failed to init alt centers buffer", K(ret));
   } else {
     for (int64_t i = 0; OB_SUCC(ret) && i < k; ++i) {
       if (OB_FAIL(centers_[0].push_back(dim, const_cast<float *>(rows + i * dim)))) {
         SHARE_LOG(WARN, "failed to push_back external center row", K(ret), K(i));
       }
     }
     if (OB_SUCC(ret)) {
       cur_idx_ = 0;
       status_ = FINISH;
     }
   }
   return ret;
 }

 int ObKmeansAlgo::inner_build(const ObIArray<float*> &input_vectors)
 {
   int ret = OB_SUCCESS;
   switch (status_) {
     case PREPARE_CENTERS: {
       center_init_start_ms_ = ObTimeUtility::current_time_ms();
       if (kmeans_ctx_->lists_ >= input_vectors.count()) {
         if (OB_FAIL(quick_centers(input_vectors))) {
           SHARE_LOG(WARN, "failed to quick centers", K(ret));
         }
       } else if (OB_FAIL(init_first_center(input_vectors))) {
         SHARE_LOG(WARN, "failed to init first center", K(ret));
       }
       break;
     }
     case INIT_CENTERS: {
       if (OB_FAIL(init_centers(input_vectors))) {
         SHARE_LOG(WARN, "failed to init centers", K(ret));
       }
       break;
     }
     case RUNNING_KMEANS: {
       if (OB_FAIL(do_kmeans(input_vectors))) {
         SHARE_LOG(WARN, "failed to do kmeans", K(ret));
       }
       break;
     }
     case FINISH: {
       LOG_INFO("finish kmeans build", K(ret));
       break;
     }
     default: {
       ret = OB_ERR_UNEXPECTED;
       SHARE_LOG(WARN, "not expected status", K(ret), K(status_));
       break;
     }
   }
   return ret;
 }
 
 // only use sample vectors as centers
 int ObKmeansAlgo::quick_centers(const ObIArray<float*> &input_vectors)
 {
   int ret = OB_SUCCESS;
   if (IS_NOT_INIT) {
     ret = OB_NOT_INIT;
     SHARE_LOG(WARN, "kmeans ctx is not inited", K(ret));
   } else if (PREPARE_CENTERS != status_) {
     ret = OB_STATE_NOT_MATCH;
     SHARE_LOG(WARN, "status not match", K(ret), K(status_));
   } else if (input_vectors.count() == 0) {
     SHARE_LOG(INFO, "input vectors is empty, skip quick centers", K(ret), K(input_vectors.count()));
   } else if (OB_FAIL(centers_[cur_idx_].init(kmeans_ctx_->dim_, input_vectors.count(), ivf_build_mem_ctx_))) {
     SHARE_LOG(WARN, "failed to init center buffer", K(ret));
   } else {
     for (int64_t i = 0; OB_SUCC(ret) && i < input_vectors.count(); ++i) {
       if (OB_FAIL(centers_[cur_idx_].push_back(kmeans_ctx_->dim_, input_vectors.at(i)))) {
         SHARE_LOG(WARN, "failed to push back center", K(ret));
       }
     }
   }
   if (OB_SUCC(ret)) {
     const int64_t center_init_cost_ms = ObTimeUtility::current_time_ms() - center_init_start_ms_;
     SHARE_LOG(INFO, "center init finished (quick_centers)", K(ret), K(center_init_cost_ms));
     kmeans_log("center_init_finished quick_centers center_init_cost_ms=%ld", center_init_cost_ms);
     status_ = FINISH;
     const int64_t center_count = centers_[cur_idx_].count();
     const int64_t sample_count = input_vectors.count();
     SHARE_LOG(INFO, "success to quick centers", K(ret), K(center_count), K(kmeans_ctx_->lists_), K(sample_count));
     kmeans_log(
         "cluster_size_stats quick_centers: no clustering assignment; centers_built=%ld samples=%ld lists=%ld",
         center_count,
         sample_count,
         kmeans_ctx_->lists_);
   }
   return ret;
 }
 
 int ObKmeansAlgo::init_first_center(const ObIArray<float *> &input_vectors)
 {
   int ret = OB_SUCCESS;
   if (PREPARE_CENTERS != status_) {
     ret = OB_STATE_NOT_MATCH;
     SHARE_LOG(WARN, "status not match", K(ret), K(status_));
   } else if (OB_FAIL(centers_[0].init(kmeans_ctx_->dim_, kmeans_ctx_->lists_, ivf_build_mem_ctx_))) {
     SHARE_LOG(WARN, "failed to init center buffer", K(ret));
   } else if (OB_FAIL(centers_[1].init(kmeans_ctx_->dim_, kmeans_ctx_->lists_, ivf_build_mem_ctx_))) {
     SHARE_LOG(WARN, "failed to init center buffer", K(ret));
   } else if (enable_parallel_) {
     // Get task handler, only get once
     ObPluginVectorIndexService *service = MTL(ObPluginVectorIndexService *);
     if (OB_ISNULL(service)) {
       ret = OB_ERR_UNEXPECTED;
       SHARE_LOG(WARN, "failed to get plugin vector index service", K(ret));
     } else {
       task_handler_ = &service->get_kmeans_build_handler();
       int64_t max_thread_cnt = 0;
 
       // Initialize task handler
       if (OB_ISNULL(task_handler_)) {
         ret = OB_ERR_UNEXPECTED;
         SHARE_LOG(WARN, "task handler is null, should be initialized in init_first_center", K(ret));
       } else if (OB_FAIL(init_build_handle(*task_handler_))) {
         SHARE_LOG(WARN, "failed to init build handle", K(ret));
       } else if (OB_FAIL(task_handler_->get_max_thread_count(max_thread_cnt, true /*with_refresh*/))) {
         SHARE_LOG(WARN, "failed to get max thread count", K(ret));
       } else {
         // Allocate task array memory, determine task count based on thread count and vector count
         const int64_t sample_cnt = input_vectors.count();
         // Ensure each task processes at least 1 vector, task count not exceeding the smaller of thread count and vector count
         max_distance_tasks_ = std::min(max_thread_cnt, sample_cnt);
         max_assign_tasks_ = max_distance_tasks_; // Assignment task count same as distance calculation tasks
 
         // Allocate distance calculation task array
         void *tmp_buf = ivf_build_mem_ctx_.Allocate(sizeof(ObKmeansDistanceCalcTask) * max_distance_tasks_);
         if (OB_ISNULL(tmp_buf)) {
           ret = OB_ALLOCATE_MEMORY_FAILED;
           SHARE_LOG(WARN, "failed to alloc distance tasks memory", K(ret));
         } else {
           distance_tasks_ = static_cast<ObKmeansDistanceCalcTask*>(tmp_buf);
           // Initialize distance calculation task objects
           for (int64_t i = 0; i < max_distance_tasks_; ++i) {
             new (&distance_tasks_[i]) ObKmeansDistanceCalcTask();
           }
         }
 
         // Allocate vector assignment task array
         if (OB_SUCC(ret)) {
           void *assign_tmp_buf = ivf_build_mem_ctx_.Allocate(sizeof(ObKmeansAssignTask) * max_assign_tasks_);
           if (OB_ISNULL(assign_tmp_buf)) {
             ret = OB_ALLOCATE_MEMORY_FAILED;
             SHARE_LOG(WARN, "failed to alloc assign tasks memory", K(ret));
           } else {
             assign_tasks_ = static_cast<ObKmeansAssignTask*>(assign_tmp_buf);
             // Initialize vector assignment task objects
             for (int64_t i = 0; i < max_assign_tasks_; ++i) {
               new (&assign_tasks_[i]) ObKmeansAssignTask();
             }
           }
         }
       }
     }
   }
 
   if (OB_SUCC(ret)) {
     const int64_t sample_cnt = input_vectors.count();
     int64_t random = 0;
     random = ObRandom::rand(0, sample_cnt - 1);
     // use random sample vector as the first center
     if (OB_FAIL(centers_[cur_idx_].push_back(kmeans_ctx_->dim_, input_vectors.at(random)))) {
       SHARE_LOG(WARN, "failed to push back center", K(ret));
     } else if (OB_ISNULL(weight_ = static_cast<float *>(ivf_build_mem_ctx_.Allocate(sizeof(float) * sample_cnt)))) {
       ret = OB_ALLOCATE_MEMORY_FAILED;
       SHARE_LOG(WARN, "failed to alloc memory", K(ret), K(ivf_build_mem_ctx_.get_all_vsag_use_mem_byte()));
     } else {
       for (int64_t i = 0; i < sample_cnt; ++i) {
         weight_[i] = FLT_MAX;
       }
       status_ = INIT_CENTERS;
       SHARE_LOG(TRACE, "success to init first center", K(ret));
     }
   }
   return ret;
 }
 
 // Kmeans++
 int ObKmeansAlgo::init_centers(const ObIArray<float*> &input_vectors)
 {
   int ret = OB_SUCCESS;
 
   if (INIT_CENTERS != status_) {
     ret = OB_STATE_NOT_MATCH;
     SHARE_LOG(WARN, "status not match", K(ret), K(status_));
   } else {
     int64_t center_idx = centers_[cur_idx_].count() - 1;
     float *current_center = centers_[cur_idx_].at(center_idx);
 
     float distance = 0;
     bool is_finish = kmeans_ctx_->lists_ == centers_[cur_idx_].count();
     float sum = 0;
 
     if (is_finish) {
       const int64_t center_init_cost_ms = ObTimeUtility::current_time_ms() - center_init_start_ms_;
       SHARE_LOG(INFO, "center init finished (kmeans++)", K(ret), K(center_init_cost_ms));
       kmeans_log("center_init_finished kmeanspp center_init_cost_ms=%ld", center_init_cost_ms);
       status_ = RUNNING_KMEANS;
       const int64_t center_count = centers_[cur_idx_].count();
       const int64_t sample_count = input_vectors.count();
       SHARE_LOG(INFO, "success to init all centers", K(ret), K(center_count), K(kmeans_ctx_->lists_), K(sample_count));
     } else if (OB_FAIL(calc_distances_parallel(input_vectors, current_center, sum))) {  // Use block parallel distance calculation
       SHARE_LOG(WARN, "failed to calc distances parallel", K(ret));
     } else {
       const int64_t sample_cnt = input_vectors.count();
       // get the next center randomly
       float random_weight = (float)ObRandom::rand(1, 100) / 100.0 * sum;
       int64_t i = 0;
       for (i = 0; i < sample_cnt; ++i) {
         if ((random_weight -= weight_[i]) <= 0.0) {
           break;
         }
       }
       if (i >= sample_cnt) {
         i = sample_cnt - 1 < 0 ? 0 : sample_cnt - 1;
       }
       if (OB_FAIL(centers_[cur_idx_].push_back(kmeans_ctx_->dim_, input_vectors.at(i)))) {
         SHARE_LOG(WARN, "failed to push back center", K(ret));
       } else {
         const int64_t center_count = centers_[cur_idx_].count();
         if (OB_NOT_NULL(kmeans_monitor_)) {
           const int64_t progress_percent = (center_count * 100) / kmeans_ctx_->lists_;
           kmeans_monitor_->set_kmeams_monitor(progress_percent, 0, 0, 0);
         }
         SHARE_LOG(TRACE, "success to init center", K(ret), K(center_count));
       }
     }
   }
   return ret;
 }
 
 int ObKmeansAlgo::calc_kmeans_distance(const float* a, const float* b, const int64_t len, float &distance)
 {
   int ret = OB_SUCCESS;
   // only use l2_distance
   distance = ObVectorL2Distance<float>::l2_square_flt_func(a, b, len);
   return ret;
 }
 
 void ObKmeansAlgo::set_centers_distance(float* centers_distance, int64_t i, int64_t j, float distance)
 {
   if (i != j) {
     (i > j) ? centers_distance[i * (i - 1) / 2 + j] = distance : centers_distance[j * (j - 1) / 2 + i] = distance;
   }
 }
 
 float ObKmeansAlgo::get_centers_distance(float* centers_distance, int64_t i, int64_t j)
 {
   if (i != j) {
     return (i > j) ? centers_distance[i * (i - 1) / 2 + j] : centers_distance[j * (j - 1) / 2 + i];
   } else {
     return 0.0;
   }
 }
 
 double ObKmeansAlgo::calc_imbalance_factor(const ObIArray<float*> &input_vectors, int32_t *data_cnt_in_cluster)
 {
   double imbalance_factor = 0.0;
   if (OB_ISNULL(data_cnt_in_cluster) || kmeans_ctx_->lists_ <= 0) {
     return imbalance_factor;
   }
 
   int64_t total_vectors = 0;
   double sum_squares = 0.0;
 
   for (int64_t i = 0; i < kmeans_ctx_->lists_; ++i) {
     total_vectors += data_cnt_in_cluster[i];
     sum_squares += static_cast<double>(data_cnt_in_cluster[i]) * data_cnt_in_cluster[i];
   }
 
   if (total_vectors > 0) {
     imbalance_factor = sum_squares * kmeans_ctx_->lists_ / (static_cast<double>(total_vectors) * total_vectors);
   }
 
   return imbalance_factor;
 }
 
 int ObKmeansAlgo::init_build_handle(ObKmeansBuildTaskHandler &handle)
 {
   int ret = OB_SUCCESS;
 
   common::ObSpinLockGuard init_guard(handle.lock_);                  // lock thread pool init to avoid init twice
   ObPluginVectorIndexService *service = MTL(ObPluginVectorIndexService *);
   if (OB_ISNULL(service)) {
     ret = OB_ERR_UNEXPECTED;
     LOG_WARN("unexpected nullptr", K(ret));
   } else if (handle.get_tg_id() != ObKmeansBuildTaskHandler::INVALID_TG_ID) {
     // no need to init twice, skip
   } else if (OB_FAIL(service->start_kmeans_tg())) {
     LOG_WARN("fail to start kmeans thread pool", K(ret));
   } else if (OB_FAIL(handle.init(service->get_kmeans_tg_id()))) {
     LOG_WARN("fail to init vector kmeans build task handle", K(ret));
   } else if (OB_FAIL(handle.start())) {
     LOG_WARN("fail to start vector kmeans build thread pool", K(ret));
   }
 
   return ret;
 }
 
 // Block parallel distance calculation
 int ObKmeansAlgo::calc_distances_parallel(const ObIArray<float*> &input_vectors,
                                          float *current_center, float &sum)
 {
   int ret = OB_SUCCESS;
   const int64_t sample_cnt = input_vectors.count();
 
   // Use cached task handler
   if (OB_ISNULL(task_handler_)) {
     // If task handler is null, fallback to serial calculation
     if (OB_FAIL(calc_distances_range(input_vectors, 0, sample_cnt, current_center, weight_, kmeans_ctx_->dim_, sum))) {
       SHARE_LOG(WARN, "failed to calc distances range", K(ret));
     }
   } else if (OB_UNLIKELY(OB_ISNULL(distance_tasks_) || max_distance_tasks_ <= 0)) {
     ret = OB_ERR_UNEXPECTED;
     SHARE_LOG(WARN, "unexpected nullptr", K(ret));
   } else {
     // NOTE: max_distance_tasks_ + 1 is used to handle the final task
     const int64_t block_size = std::max(1L, sample_cnt / (max_distance_tasks_ + 1));
     int64_t end_idx = 0;
 
     // Create and submit tasks
     for (int64_t i = 0; OB_SUCC(ret) && i < max_distance_tasks_; ++i) {
       // max_distance_tasks_ is min(sample_cnt, max_thread_cnt), so end_idx <= sample_cnt
       int64_t start_idx = i * block_size;
       end_idx = start_idx + block_size;
 
       ObKmeansDistanceCalcTask &task = distance_tasks_[i];
       // Reset task state to ensure proper reuse
       task.reset();
       if (OB_FAIL(task.init(start_idx, end_idx, &input_vectors, current_center, weight_, kmeans_ctx_->dim_))) {
         SHARE_LOG(WARN, "failed to init distance calc task", K(ret), K(i));
       } else if (OB_FAIL(task_handler_->push_task(task))) {
         if (OB_EAGAIN != ret) {
           SHARE_LOG(WARN, "failed to push distance calc task", K(ret), K(i));
         } else if (check_stop()) {  // check_stop in each loop will lead to a decrease in performance; the decision is
                                     // only made in exceptional cases.
           ret = OB_CANCELED;
           SHARE_LOG(WARN, "check stop", K(ret));
         } else if (OB_FAIL(task.do_work())) {
           SHARE_LOG(WARN, "failed to do distance calc task", K(ret));
         }
       }
     }
 
     // Handle the final task
     if (OB_SUCC(ret)) {
       float tmp_sum = 0.0f;
       if (OB_FAIL(calc_distances_range(input_vectors, end_idx, sample_cnt, current_center, weight_, kmeans_ctx_->dim_,
                                        tmp_sum))) {
         SHARE_LOG(WARN, "failed to calc distances range", K(ret));
       } else {
         sum += tmp_sum;
       }
     } else {
       for (int64_t i = 0; i < max_distance_tasks_; ++i) {
         ObKmeansDistanceCalcTask &task = distance_tasks_[i];
         task.set_task_stop();
       }
     }
     // Note. Anywhere above, all tasks need to be stopped otherwise it may cause a core dump because sample vectors
     // will be released.
     wait_parallel_task_finish(distance_tasks_, max_distance_tasks_, *task_handler_);
 
     // Wait for all tasks to complete and collect results
     for (int64_t i = 0; OB_SUCC(ret) && i < max_distance_tasks_; ++i) {
       ObKmeansDistanceCalcTask &task = distance_tasks_[i];
       if (OB_FAIL(task.get_ret())) {
         SHARE_LOG(WARN, "distance calc task failed", K(ret), K(i));
       } else {
         sum += task.get_sum();
       }
     }
   }
 
   return ret;
 }
 
 // ------------------ ObKmeansExecutor implement ------------------
 
 bool ObKmeansExecutor::check_stop()
 {
   int ret = OB_SUCCESS;
   if (OB_FAIL(share::dag_yield())) {
     LOG_WARN("dag yield failed", K(ret)); // exit for dag task as soon as possible after canceled.
   } else if (OB_FAIL(THIS_WORKER.check_status())) {
     LOG_WARN("check status failed", K(ret));
   }
   return OB_SUCCESS != ret;
 }
 
 int ObKmeansExecutor::append_sample_vector(float* vector)
 {
   int ret = OB_SUCCESS;
   if (IS_NOT_INIT) {
     ret = OB_NOT_INIT;
     SHARE_LOG(WARN, "kmeans ctx is not inited", K(ret));
   } else if (OB_ISNULL(vector)) {
     ret = OB_INVALID_ARGUMENT;
     SHARE_LOG(WARN, "invalid null vector", K(ret));
   } else if (OB_FAIL(ctx_.append_sample_vector(vector))) {
     LOG_WARN("failed to append sample vector", K(ret), K(ctx_));
   } else {
     ++ctx_.total_scan_count_;
   }
   return ret;
 }
 
 // ------------------ ObSingleKmeansExecutor implement ------------------
 int ObSingleKmeansExecutor::init(
     ObKmeansAlgoType algo_type,
     const int64_t tenant_id,
     const int64_t lists,
     const int64_t samples_per_nlist,
     const int64_t dim,
     ObVectorIndexDistAlgorithm dist_algo,
     ObVectorNormalizeInfo *norm_info/* = nullptr*/,
     const int64_t pq_m_size/* = 1*/)
 {
   int ret = OB_SUCCESS;
   if (OB_FAIL(ctx_.init(tenant_id, lists, samples_per_nlist, dim, dist_algo, norm_info, 1 /*pq_m*/, false /* is_pq_stage */))) {
     LOG_WARN("fail to init kmeans ctx", K(ret), K(tenant_id), K(lists), K(samples_per_nlist), K(dim), K(dist_algo));
   } else {
     if (algo_type == ObKmeansAlgoType::KAT_EXTERNAL_GPU) {
       ctx_.set_train_strategy(KTS_FULL_BATCH);
       kmeans_log("external_kmeans: train_strategy set to FULL_BATCH (external worker does not support NMBKM)");
     }
     if (algo_type == ObKmeansAlgoType::KAT_ELKAN) {
       void *tmp_buf = nullptr;
       if (OB_ISNULL(tmp_buf = ivf_build_mem_ctx_.Allocate(sizeof(ObElkanKmeansAlgo)))) {
         ret = OB_ALLOCATE_MEMORY_FAILED;
         LOG_WARN("failed to alloc tmp_buf", K(ret), K(ivf_build_mem_ctx_.get_all_vsag_use_mem_byte()));
       } else {
         algo_ = new (tmp_buf) ObElkanKmeansAlgo(ivf_build_mem_ctx_);
       }
     } else if (algo_type == ObKmeansAlgoType::KAT_EXTERNAL_GPU) {
       void *tmp_buf = nullptr;
       if (OB_ISNULL(tmp_buf = ivf_build_mem_ctx_.Allocate(sizeof(ObExternalGpuKmeansAlgo)))) {
         ret = OB_ALLOCATE_MEMORY_FAILED;
         LOG_WARN("failed to alloc tmp_buf for external kmeans", K(ret), K(ivf_build_mem_ctx_.get_all_vsag_use_mem_byte()));
       } else {
         algo_ = new (tmp_buf) ObExternalGpuKmeansAlgo(ivf_build_mem_ctx_);
       }
     } else {
       ret = OB_INVALID_ARGUMENT;
       LOG_WARN("invalid kmeans algorithm type", K(ret), K(algo_type));
     }
   }
 
   if (FAILEDx(algo_->init(ctx_, true /* enable_parallel */))) {
     LOG_WARN("fail to init kmeans algo", K(ret), K(ctx_));
   } else {
     is_inited_ = true;
   }
   return ret;
 }
 
 int ObSingleKmeansExecutor::build(ObInsertMonitor *insert_monitor)
 {
   int ret = OB_SUCCESS;
   int64_t start_time = ObTimeUtil::current_time_ms();
   if (IS_NOT_INIT) {
     ret = OB_NOT_INIT;
     SHARE_LOG(WARN, "kmeans ctx is not inited", K(ret));
   } else if (OB_NOT_NULL(insert_monitor) && OB_FALSE_IT(algo_->set_kmeans_monitor(insert_monitor->kmeans_monitor_))) {
   } else if (OB_FAIL(ctx_.try_normalize_samples())) {
     LOG_WARN("fail to try normalize all samples", K(ret), K(ctx_));
   } else if (OB_FAIL(algo_->build(ctx_.sample_vectors_))) {
     LOG_WARN("fail to build kmeans algo", K(ret), K(algo_));
   } else if (OB_NOT_NULL(insert_monitor)) {
     insert_monitor->kmeans_monitor_.add_finish_tablet_cnt();
   }
   if (OB_NOT_NULL(algo_)) {
     algo_->destroy();
   }
   LOG_INFO("SingleKmeans build cost", K(ret), K(ObTimeUtil::current_time_ms() - start_time));
   return ret;
 }
 
 int ObSingleKmeansExecutor::get_kmeans_algo(ObKmeansAlgo *&algo)
 {
   int ret = OB_SUCCESS;
   if (IS_NOT_INIT) {
     ret = OB_NOT_INIT;
     SHARE_LOG(WARN, "kmeans ctx is not inited", K(ret));
   } else if (OB_ISNULL(algo_)) {
     ret = OB_ERR_NULL_VALUE;
     LOG_WARN("invalid null algo", K(ret));
   } else {
     algo = algo_;
   }
   return ret;
 }
 
 int64_t ObSingleKmeansExecutor::get_centers_count() const
 {
   int64_t i_ret = 0;
   if (OB_NOT_NULL(algo_)) {
     i_ret = algo_->get_cur_centers().count();
   }
   return i_ret;
 }
 
 int64_t ObSingleKmeansExecutor::get_centers_dim() const
 {
   int64_t i_ret = 0;
   if (OB_NOT_NULL(algo_)) {
     i_ret = algo_->get_cur_centers().dim_;
   }
   return i_ret;
 }
 
 int ObSingleKmeansExecutor::get_center(const int64_t pos, float *&center_vector)
 {
   int ret = OB_SUCCESS;
   if (pos < 0 || pos >= get_centers_count()) {
     ret = OB_INDEX_OUT_OF_RANGE;
     LOG_WARN("index out of range", K(ret), K(pos), K(get_centers_count()));
   } else {
     center_vector = algo_->get_cur_centers().at(pos);
   }
   return ret;
 }
 
 // ------------------ ObMultiKmeansExecutor implement ------------------
 ObMultiKmeansExecutor::~ObMultiKmeansExecutor()
 {
   for (int i = 0; i < algos_.count(); ++i) {
     if (OB_NOT_NULL(algos_[i])) {
       algos_[i]->~ObKmeansAlgo();
       ivf_build_mem_ctx_.Deallocate(algos_[i]);
       algos_[i] = nullptr;
     }
   }
   algos_.reset();
 }
 
 int ObMultiKmeansExecutor::init(
     ObKmeansAlgoType algo_type,
     const int64_t tenant_id,
     const int64_t lists,
     const int64_t samples_per_nlist,
     const int64_t dim,
     ObVectorIndexDistAlgorithm dist_algo,
     ObVectorNormalizeInfo *norm_info/* = nullptr*/,
     const int64_t pq_m_size/* = 1*/)
 {
   int ret = OB_SUCCESS;
   algo_type_ = algo_type;
   use_external_pq_batch_ = false;
   if (OB_FAIL(ctx_.init(tenant_id, lists, samples_per_nlist, dim, dist_algo, norm_info, pq_m_size, true /* is_pq_stage */))) {
     LOG_WARN("fail to init kmeans ctx", K(ret), K(tenant_id), K(lists), K(samples_per_nlist), K(dim), K(dist_algo));
   } else {
     if (algo_type == ObKmeansAlgoType::KAT_EXTERNAL_GPU) {
       ctx_.set_train_strategy(KTS_FULL_BATCH);
       kmeans_log("external_kmeans: train_strategy set to FULL_BATCH (external worker does not support NMBKM)");
       if (pq_m_size > 1 && ob_external_pq_batch_enabled_by_env()) {
         use_external_pq_batch_ = true;
         kmeans_log(
             "external_kmeans: PQ batch worker enabled m=%ld micro_batch=%d (OB_EXTERNAL_KMEANS_PQ_BATCH=0 to disable)",
             pq_m_size,
             ob_external_pq_micro_batch_from_env());
       }
     }
     pq_m_size_ = pq_m_size;
     if (OB_FAIL(algos_.prepare_allocate(pq_m_size))) {
       LOG_WARN("fail to reserve space", K(ret), K(pq_m_size));
     }
 
     for (int i = 0; OB_SUCC(ret) && i < algos_.count(); ++i) {
       if (algo_type == ObKmeansAlgoType::KAT_ELKAN) {
         void *tmp_buf = nullptr;
         if (OB_ISNULL(tmp_buf = ivf_build_mem_ctx_.Allocate(sizeof(ObElkanKmeansAlgo)))) {
           ret = OB_ALLOCATE_MEMORY_FAILED;
           LOG_WARN("failed to alloc tmp_buf", K(ret), K(ivf_build_mem_ctx_.get_all_vsag_use_mem_byte()));
         } else {
           algos_[i] = new (tmp_buf) ObElkanKmeansAlgo(ivf_build_mem_ctx_);
         }
       } else if (algo_type == ObKmeansAlgoType::KAT_EXTERNAL_GPU) {
         void *tmp_buf = nullptr;
         if (OB_ISNULL(tmp_buf = ivf_build_mem_ctx_.Allocate(sizeof(ObExternalGpuKmeansAlgo)))) {
           ret = OB_ALLOCATE_MEMORY_FAILED;
           LOG_WARN("failed to alloc tmp_buf for external kmeans", K(ret), K(ivf_build_mem_ctx_.get_all_vsag_use_mem_byte()));
         } else {
           algos_[i] = new (tmp_buf) ObExternalGpuKmeansAlgo(ivf_build_mem_ctx_);
         }
       } else {
         ret = OB_INVALID_ARGUMENT;
         LOG_WARN("invalid kmeans algorithm type", K(ret), K(algo_type));
       }
       if (FAILEDx(algos_[i]->init(ctx_))) {
         LOG_WARN("fail to init kmeans algo", K(ret), K(ctx_));
       }
     }
   }
 
   if (OB_SUCC(ret)) {
     is_inited_ = true;
   }
   return ret;
 }
 
 int ObMultiKmeansExecutor::build(ObInsertMonitor *insert_monitor)
 {
   int ret = OB_SUCCESS;
   if (IS_NOT_INIT) {
     ret = OB_NOT_INIT;
     SHARE_LOG(WARN, "kmeans ctx is not inited", K(ret));
   } else if (use_external_pq_batch_) {
     if (OB_FAIL(build_external_pq_batch(insert_monitor))) {
       LOG_WARN("fail to build external pq batch", K(ret));
     }
   } else {
     int64_t start_time = ObTimeUtil::current_time_ms();
     ObArenaAllocator tmp_alloc("MulKmeans", OB_MALLOC_NORMAL_BLOCK_SIZE, ctx_.tenant_id_);
     // init spilited_arrs, size: m * sample_vectors_.count()
     ObArrayArray<float*> splited_arrs(OB_MALLOC_NORMAL_BLOCK_SIZE, ModulePageAllocator(tmp_alloc, "MulKmeans"));
     if (OB_FAIL(prepare_splited_arrs(splited_arrs))) {
       LOG_WARN("fail to prepare splited_arrs", K(ret));
     } else if (OB_NOT_NULL(insert_monitor)) {
       if (OB_NOT_NULL(insert_monitor->kmeans_monitor_.vec_index_task_total_cnt_)) {
         (void)ATOMIC_AAF(insert_monitor->kmeans_monitor_.vec_index_task_total_cnt_, pq_m_size_);
       }
       if (OB_NOT_NULL(insert_monitor->kmeans_monitor_.vec_index_task_thread_pool_cnt_)) {
         (void)ATOMIC_SET(insert_monitor->kmeans_monitor_.vec_index_task_thread_pool_cnt_, 1);
       }
     }
     for (int i = 0; OB_SUCC(ret) && i < pq_m_size_; ++i) {
       if (i >= algos_.count()) {
         ret = OB_ERR_UNEXPECTED;
         LOG_WARN("size of algos_ should be pq_m_size_", K(ret), K(i), K(pq_m_size_), K(algos_.count()));
       } else if (OB_FAIL(algos_[i]->build(splited_arrs.at(i)))) {
         LOG_WARN("fail to build kmeans algo", K(ret), K(i), K(algos_[i]));
       } else if (OB_NOT_NULL(insert_monitor) && OB_NOT_NULL(insert_monitor->kmeans_monitor_.vec_index_task_finish_cnt_)) {
         (void)ATOMIC_AAF(insert_monitor->kmeans_monitor_.vec_index_task_finish_cnt_, 1);
       }
       if (OB_NOT_NULL(algos_[i])) {
         algos_[i]->destroy();
       }
     }
     if (OB_SUCC(ret) && OB_NOT_NULL(insert_monitor)) {
       insert_monitor->kmeans_monitor_.add_finish_tablet_cnt();
     }
     LOG_INFO("MultiKmeans build cost", K(ret), K(ObTimeUtil::current_time_ms() - start_time));
   }
 
   return ret;
 }

 int ObMultiKmeansExecutor::build_external_pq_batch(ObInsertMonitor *insert_monitor)
 {
   int ret = OB_SUCCESS;
   char in_template[] = "/tmp/ob_ext_km_pq_in_XXXXXX";
   char out_template[] = "/tmp/ob_ext_km_pq_out_XXXXXX";
   int in_fd = ::mkstemp(in_template);
   int out_fd = ::mkstemp(out_template);
   const int64_t wall_t0_ms = ObTimeUtility::current_time_ms();
   if (IS_NOT_INIT) {
     ret = OB_NOT_INIT;
     SHARE_LOG(WARN, "kmeans ctx is not inited", K(ret));
   } else if (in_fd < 0 || out_fd < 0) {
     ret = OB_IO_ERROR;
     SHARE_LOG(WARN, "mkstemp pq batch failed", K(ret), K(in_fd), K(out_fd));
   } else {
     ::close(out_fd);
     out_fd = -1;
     ::close(in_fd);
     in_fd = -1;
     const char *cmd = ob_get_external_kmeans_pq_batch_cmd_or_default();
     const int max_iters = ob_external_kmeans_max_iters_from_env();
     const int micro_batch = ob_external_pq_micro_batch_from_env();
     const int64_t n = ctx_.sample_vectors_.count();
     const int64_t full_dim = ctx_.sample_dim_;
     const int64_t sub_dim = ctx_.dim_;
     const int64_t k = ctx_.lists_;
     const int64_t m = pq_m_size_;
     const bool gpu_kpp = ob_external_gpu_kmeanspp_enabled();
     if (OB_ISNULL(cmd) || cmd[0] == '\0') {
       ret = OB_ERR_UNEXPECTED;
       kmeans_log("external_pq_batch_fail reason=cmd_empty");
     } else if (n <= 0 || full_dim <= 0 || sub_dim <= 0 || k <= 0 || m <= 0) {
       ret = OB_INVALID_ARGUMENT;
       kmeans_log("external_pq_batch_fail reason=invalid_dims n=%ld full_dim=%ld m=%ld sub_dim=%ld k=%ld",
           n,
           full_dim,
           m,
           sub_dim,
           k);
     } else if (full_dim != m * sub_dim) {
       ret = OB_INVALID_ARGUMENT;
       kmeans_log("external_pq_batch_fail reason=full_dim_mismatch full_dim=%ld m=%ld sub_dim=%ld",
           full_dim,
           m,
           sub_dim);
     } else if (pq_m_size_ != algos_.count()) {
       ret = OB_ERR_UNEXPECTED;
       LOG_WARN("algos count mismatch", K(ret), K(pq_m_size_), K(algos_.count()));
     } else {
       const int64_t write_t0_ms = ObTimeUtility::current_time_ms();
       if (OB_FAIL(write_ob_external_kmeans_pq_batch_input(
                      in_template,
                      ctx_.sample_vectors_,
                      full_dim,
                      m,
                      sub_dim,
                      k,
                      max_iters,
                      micro_batch,
                      gpu_kpp))) {
         SHARE_LOG(WARN, "write pq batch input failed", K(ret));
         kmeans_log("external_pq_batch_fail reason=write_input ret=%d", ret);
       } else {
         kmeans_log(
             "external_pq_batch_invoke n=%ld m=%ld sub_dim=%ld k=%ld max_iters=%d micro_batch=%d gpu_kpp=%d "
             "write_ms=%ld in=%s out=%s",
             n,
             m,
             sub_dim,
             k,
             max_iters,
             micro_batch,
             gpu_kpp ? 1 : 0,
             ObTimeUtility::current_time_ms() - write_t0_ms,
             in_template,
             out_template);
         if (OB_ISNULL(::getenv("USE_FLASH_KMEANS")) && OB_NOT_NULL(std::strstr(cmd, "flash-kmeans"))) {
           (void)::setenv("USE_FLASH_KMEANS", "1", 0);
         }
         char cmdline[2048];
         const int ncmd = snprintf(cmdline, sizeof(cmdline), "%s \"%s\" \"%s\"", cmd, in_template, out_template);
         if (ncmd <= 0 || ncmd >= static_cast<int>(sizeof(cmdline))) {
           ret = OB_ERR_UNEXPECTED;
           kmeans_log("external_pq_batch_fail reason=cmdline_too_long n=%d", ncmd);
         } else {
           const int64_t sys_t0_ms = ObTimeUtility::current_time_ms();
           const int sys_ret = ::system(cmdline);
           kmeans_log(
               "external_pq_batch_system_return sys_ret=%d system_elapsed_ms=%ld",
               sys_ret,
               ObTimeUtility::current_time_ms() - sys_t0_ms);
           if (sys_ret != 0) {
             ret = OB_ERR_UNEXPECTED;
             SHARE_LOG(WARN, "pq batch worker failed", K(ret), K(sys_ret), K(cmdline));
           }
         }
       }
     }

     if (OB_SUCC(ret)) {
       FILE *fp = fopen(out_template, "rb");
       if (OB_ISNULL(fp)) {
         ret = OB_IO_ERROR;
         kmeans_log("external_pq_batch_fail reason=fopen_output");
       } else {
         ObExtKmeansPqBatchOutHeader hdr;
         if (fread(&hdr, sizeof(hdr), 1, fp) != 1) {
           ret = OB_IO_ERROR;
           kmeans_log("external_pq_batch_fail reason=fread_output_header");
         } else if (hdr.magic_ != OB_EXT_KMEANS_PQ_BATCH_MAGIC || hdr.version_ != OB_EXT_KMEANS_PQ_BATCH_VERSION) {
           ret = OB_ERR_UNEXPECTED;
           kmeans_log("external_pq_batch_fail reason=bad_output_magic ver=%u", hdr.version_);
         } else if (hdr.m_ != m || hdr.k_ != k || hdr.sub_dim_ != sub_dim) {
           ret = OB_ERR_UNEXPECTED;
           kmeans_log("external_pq_batch_fail reason=output_shape_mismatch out_m=%ld out_k=%ld out_sub=%ld",
               hdr.m_,
               hdr.k_,
               hdr.sub_dim_);
         } else {
           const size_t row_floats = static_cast<size_t>(sub_dim);
           const size_t block_floats = static_cast<size_t>(k) * row_floats;
           std::vector<float> row_buf(block_floats);
           for (int64_t si = 0; OB_SUCC(ret) && si < m; ++si) {
             if (fread(row_buf.data(), sizeof(float), block_floats, fp) != block_floats) {
               ret = OB_IO_ERROR;
               kmeans_log("external_pq_batch_fail reason=fread_subspace si=%ld", si);
             } else if (OB_ISNULL(algos_.at(si))) {
               ret = OB_ERR_UNEXPECTED;
             } else if (OB_FAIL(algos_.at(si)->load_finished_centers(row_buf.data(), k, sub_dim))) {
               LOG_WARN("load_finished_centers failed", K(ret), K(si));
             } else if (OB_NOT_NULL(insert_monitor) &&
                        OB_NOT_NULL(insert_monitor->kmeans_monitor_.vec_index_task_finish_cnt_)) {
               (void)ATOMIC_AAF(insert_monitor->kmeans_monitor_.vec_index_task_finish_cnt_, 1);
             }
             if (OB_NOT_NULL(algos_.at(si))) {
               algos_.at(si)->destroy();
             }
           }
         }
         (void)fclose(fp);
       }
     }
     (void)::unlink(in_template);
     (void)::unlink(out_template);
   }
   if (in_fd >= 0) {
     ::close(in_fd);
   }
   if (out_fd >= 0) {
     ::close(out_fd);
   }
   if (OB_NOT_NULL(insert_monitor)) {
     if (OB_NOT_NULL(insert_monitor->kmeans_monitor_.vec_index_task_total_cnt_)) {
       (void)ATOMIC_AAF(insert_monitor->kmeans_monitor_.vec_index_task_total_cnt_, pq_m_size_);
     }
     if (OB_NOT_NULL(insert_monitor->kmeans_monitor_.vec_index_task_thread_pool_cnt_)) {
       (void)ATOMIC_SET(insert_monitor->kmeans_monitor_.vec_index_task_thread_pool_cnt_, 1);
     }
     if (OB_SUCC(ret)) {
       insert_monitor->kmeans_monitor_.add_finish_tablet_cnt();
     }
   }
   if (OB_SUCC(ret)) {
     kmeans_log(
         "external_pq_batch_done n=%ld m=%ld sub_dim=%ld k=%ld wall_elapsed_ms=%ld",
         ctx_.sample_vectors_.count(),
         pq_m_size_,
         ctx_.dim_,
         ctx_.lists_,
         ObTimeUtility::current_time_ms() - wall_t0_ms);
   }
   LOG_INFO("MultiKmeans external pq batch cost",
       K(ret),
       K(ObTimeUtility::current_time_ms() - wall_t0_ms));
   return ret;
 }

 int ObMultiKmeansExecutor::prepare_splited_arrs(ObArrayArray<float *> &splited_arrs)
 {
   int ret = OB_SUCCESS;
   const ObIArray<float *> &sample_arr = ctx_.sample_vectors_;
   if (OB_FAIL(splited_arrs.reserve(pq_m_size_))) {
     LOG_WARN("fail to prepare alloc space", K(ret), K(pq_m_size_));
   }
   for (int i = 0; OB_SUCC(ret) && i < pq_m_size_; ++i) {
     if (OB_FAIL(splited_arrs.push_back(ObArray<float *>()))) {
       LOG_WARN("fail to push back empty array", K(ret), K(i), K(pq_m_size_), K(splited_arrs.count()));
     } else if (OB_FAIL(splited_arrs.at(i).reserve(sample_arr.count()))) {
       LOG_WARN("fail to reserve space", K(ret), K(i), K(pq_m_size_), K(splited_arrs.count()));
     }
   }
 
   for (int i = 0; OB_SUCC(ret) && i < sample_arr.count(); ++i) {
     if (OB_FAIL(split_vector(sample_arr.at(i), splited_arrs))) {
       LOG_WARN("fail to split vector", K(ret));
     }
   }
 
   return ret;
 }
 
 int ObMultiKmeansExecutor::init_build_handle(ObKmeansBuildTaskHandler &handle)
 {
   int ret = OB_SUCCESS;
 
   common::ObSpinLockGuard init_guard(handle.lock_);                  // lock thread pool init to avoid init twice
   ObPluginVectorIndexService *service = MTL(ObPluginVectorIndexService *);
   if (OB_ISNULL(service)) {
     ret = OB_ERR_UNEXPECTED;
     LOG_WARN("unexpected nullptr", K(ret));
   } else if (handle.get_tg_id() != ObKmeansBuildTaskHandler::INVALID_TG_ID) {
     // no need to init twice, skip
   } else if (OB_FAIL(service->start_kmeans_tg())) {
     LOG_WARN("fail to start kmeans thread pool", K(ret));
   } else if (OB_FAIL(handle.init(service->get_kmeans_tg_id()))) {
     LOG_WARN("fail to init vector kmeans build task handle", K(ret));
   } else if (OB_FAIL(handle.start())) {
     LOG_WARN("fail to start vector kmeans build thread pool", K(ret));
   }
 
   return ret;
 }
 
 void ObMultiKmeansExecutor::wait_kmeans_task_finish(ObKmeansBuildTask *build_tasks, ObKmeansBuildTaskHandler &handle)
 {
   int ret = OB_SUCCESS;
 
   bool is_all_finish = false;
   if (OB_NOT_NULL(build_tasks)) {
     while (!is_all_finish && handle.get_task_ref() > 0) {
       if (check_stop()) {
         for (int i = 0; i < pq_m_size_; i++) {
           build_tasks[i].set_task_stop();
         }
         LOG_INFO("kmeans executor is fore stop", K(*this));
         // do not break, wait for all task finish
       }
       int64_t max_thread_cnt = 0;
       if (OB_FAIL(handle.get_max_thread_count(max_thread_cnt, true /*with_refresh*/))) {
         LOG_WARN("fail to get max thread count", K(ret));
       }
       ob_usleep(ObKmeansBuildTaskHandler::WAIT_RETRY_PUSH_TASK_TIME);
       is_all_finish = true;
       for (int i = 0; i < pq_m_size_; i++) {
         if (!build_tasks[i].is_finish()) {
           is_all_finish = false;
           break;
         }
       }
     }
   }
 }
 
 int ObMultiKmeansExecutor::do_build_task_local(const common::ObTableID &table_id, const common::ObTabletID &tablet_id,
                                                ObKmeansBuildTaskHandler &handle,
                                                const ObArrayArray<float *> &splited_arrs,
                                                ObKmeansBuildTask *build_tasks, int task_idx,
                                                ObInsertMonitor *insert_monitor)
 {
   int ret = OB_SUCCESS;
   if (OB_UNLIKELY(OB_ISNULL(build_tasks))) {
     ret = OB_INVALID_ARGUMENT;
     LOG_WARN("invalid argument", KR(ret), K(build_tasks), K(task_idx));
   }
   for (; task_idx < pq_m_size_ && OB_SUCC(ret); task_idx++) {
     if (check_stop()) {
       ret = OB_CANCELED;
       LOG_INFO("kmeans executor is fore stop", K(*this));
       break;
     }
     if (OB_NOT_NULL(insert_monitor) && OB_NOT_NULL(insert_monitor->kmeans_monitor_.vec_index_task_thread_pool_cnt_)) {
       int64_t thread_cnt = TG_GET_THREAD_CNT(handle.get_tg_id());
       (void)ATOMIC_SET(insert_monitor->kmeans_monitor_.vec_index_task_thread_pool_cnt_, thread_cnt + 1);
     }
     ObKmeansBuildTask &build_task = build_tasks[task_idx];
     int64_t max_thread_cnt = 0;
     if (OB_FAIL(build_task.init(table_id, tablet_id, task_idx, algos_[task_idx],
                                 &splited_arrs.at(task_idx), insert_monitor))) {  // 1. make task
       LOG_WARN("fail to init opt async task", KR(ret));
     } else if (OB_FAIL(handle.get_max_thread_count(max_thread_cnt, true /*with_refresh*/))) {
       LOG_WARN("fail to get max thread count", K(ret));
     } else if (handle.get_task_ref() < max_thread_cnt) {
       if (OB_FAIL(handle.push_task(build_task))) {
         LOG_WARN("fail to push build task", KR(ret), K(max_thread_cnt));
       }
     } else if (OB_FAIL(build_task.do_work())) {
       LOG_WARN("fail to do build task", KR(ret));
     }
   }
   return ret;
 }
 
 int ObMultiKmeansExecutor::build_parallel(const common::ObTableID &table_id, const common::ObTabletID &tablet_id, ObInsertMonitor* insert_monitor)
 {
   int ret = OB_SUCCESS;
   if (IS_NOT_INIT) {
     ret = OB_NOT_INIT;
     LOG_WARN("kmeans ctx is not inited", K(ret));
   } else if (use_external_pq_batch_) {
     if (OB_FAIL(build_external_pq_batch(insert_monitor))) {
       LOG_WARN("fail to build external pq batch", K(ret));
     }
   } else {
     int64_t start_time = ObTimeUtil::current_time_ms();
     LOG_INFO("start build_parallel", K(table_id), K(tablet_id), K(ctx_));
     ObArenaAllocator tmp_alloc("MulKmeans", OB_MALLOC_NORMAL_BLOCK_SIZE, ctx_.tenant_id_);
     // init spilited_arrs, size: m * sample_vectors_.count()
     ObArrayArray<float *> splited_arrs(OB_MALLOC_NORMAL_BLOCK_SIZE, ModulePageAllocator(tmp_alloc, "MulKmeans"));
     if (OB_FAIL(prepare_splited_arrs(splited_arrs))) {
       LOG_WARN("fail to prepare splited_arrs", K(ret));
     } else {
       // build_parallel
       ObPluginVectorIndexService *service = MTL(ObPluginVectorIndexService *);
       if (OB_ISNULL(service)) {
         ret = OB_ERR_UNEXPECTED;
         LOG_WARN("unexpected nullptr", K(ret));
       } else {
         ObKmeansBuildTaskHandler &handle = service->get_kmeans_build_handler();
         void *buf = nullptr;
         ObKmeansBuildTask *build_tasks = nullptr;
         int64_t max_thread_cnt = 0;
 
         if (OB_FAIL(init_build_handle(handle))) {
           LOG_WARN("fail to init build handle", K(ret));
         } else if (OB_FAIL(handle.get_max_thread_count(max_thread_cnt, true /*with_refresh*/))) {
           LOG_WARN("fail to get max thread count", K(ret));
         } else if (OB_ISNULL(buf = tmp_alloc.alloc(sizeof(ObKmeansBuildTask) * pq_m_size_))) {
           ret = OB_ALLOCATE_MEMORY_FAILED;
           LOG_WARN("fail to alloc memory of ObKmeansBuildTask", K(ret));
         } else if (FALSE_IT(build_tasks = new (buf) ObKmeansBuildTask[pq_m_size_])) {
         } else if (pq_m_size_ != algos_.count()) {
           ret = OB_ERR_UNEXPECTED;
           LOG_WARN("size of algos_ should be pq_m_size_", K(ret), K(pq_m_size_), K(algos_.count()));
         } else if (pq_m_size_ != splited_arrs.count()) {
           ret = OB_ERR_UNEXPECTED;
           LOG_WARN("size of splited_arrs should be pq_m_size_", K(ret), K(pq_m_size_), K(splited_arrs.count()));
         } else if (OB_NOT_NULL(insert_monitor) && OB_NOT_NULL(insert_monitor->kmeans_monitor_.vec_index_task_total_cnt_)) {
           (void)ATOMIC_AAF(insert_monitor->kmeans_monitor_.vec_index_task_total_cnt_, pq_m_size_);
         } else if (OB_NOT_NULL(insert_monitor) && OB_NOT_NULL(insert_monitor->kmeans_monitor_.vec_index_task_thread_pool_cnt_)) {
           int64_t thread_cnt = TG_GET_THREAD_CNT(handle.get_tg_id());
           (void)ATOMIC_SET(insert_monitor->kmeans_monitor_.vec_index_task_thread_pool_cnt_, thread_cnt);
         }
 
         int64_t a_thread_task_cnt = pq_m_size_ / (max_thread_cnt + 1);
         int task_idx = 0;
         for (; task_idx < pq_m_size_ - a_thread_task_cnt && OB_SUCC(ret); task_idx++) {
           if (check_stop()) {
             ret = OB_CANCELED;
             LOG_INFO("kmeans executor is fore stop", K(*this));
             break;
           }
           ObKmeansBuildTask &build_task = build_tasks[task_idx];
           if (OB_FAIL(build_task.init(table_id, tablet_id, task_idx, algos_[task_idx], &splited_arrs.at(task_idx), insert_monitor))) {  // 1. make task
             LOG_WARN("fail to init opt async task", KR(ret));
           } else if (OB_FAIL(handle.push_task(build_task))) {  // 2. push task
             if (OB_EAGAIN != ret) {
               SHARE_LOG(WARN, "failed to push build task", K(ret), K(task_idx));
             } else if (OB_FAIL(build_task.do_work())) {  // Degraded to single-threaded processing
               LOG_WARN("fail to do build task", KR(ret));
             }
           }
         }
 
         if (OB_SUCC(ret)) {
           // The remaining tasks are executed by the current thread.
           if (OB_FAIL(do_build_task_local(table_id, tablet_id, handle, splited_arrs, build_tasks, task_idx, insert_monitor))) {
             LOG_WARN("fail to do build task local", KR(ret));
           }
         } else {
           // Note. If the previous process fails, all tasks need to be stopped otherwise it may cause a core dump
           for (int i = 0; i < pq_m_size_; i++) {
             ObKmeansBuildTask &build_task = build_tasks[i];
             build_task.set_task_stop();
           }
         }
 
         // 3. wait for all task finish
         wait_kmeans_task_finish(build_tasks, handle);
 
         // 4. check task result
         for (int i = 0; i < pq_m_size_ && OB_SUCC(ret); i++) {
           if (OB_UNLIKELY(OB_ISNULL(build_tasks))) {
             ret = OB_ERR_UNEXPECTED;
             LOG_WARN("unexpected nullptr", K(ret));
           } else if (OB_FAIL(build_tasks[i].get_ret())) {
             LOG_WARN("fail to build kmeans algo", K(ret), K(i), K(build_tasks[i]));
           }
         }  // end for
         if (OB_SUCC(ret) && OB_NOT_NULL(insert_monitor)) {
           insert_monitor->kmeans_monitor_.add_finish_tablet_cnt();
         }
       }
     }
     LOG_INFO("build_parallel cost", K(ret), K(ObTimeUtil::current_time_ms() - start_time));
   }
 
   return ret;
 }
 
 int ObMultiKmeansExecutor::split_vector(float *vector, ObArrayArray<float *> &splited_arrs)
 {
   int ret = OB_SUCCESS;
   int64_t start_idx = 0;
   for (int i = 0; OB_SUCC(ret) && i < pq_m_size_; ++i) {
     if (OB_FAIL(splited_arrs.push_back(i, vector + start_idx))) {
       SHARE_LOG(WARN, "failed to push back array", K(ret), K(i));
     } else {
       start_idx += ctx_.dim_;
     }
   }
   return ret;
 }
 
 int64_t ObMultiKmeansExecutor::get_total_centers_count() const
 {
   int64_t i_ret = 0;
   if (!algos_.empty() && OB_NOT_NULL(algos_[0])) {
     i_ret = algos_[0]->get_cur_centers().count() * algos_.count();
   }
   return i_ret;
 }
 
 int64_t ObMultiKmeansExecutor::get_centers_count_per_kmeans() const
 {
   int64_t i_ret = 0;
   if (!algos_.empty() && OB_NOT_NULL(algos_[0])) {
     i_ret = algos_[0]->get_cur_centers().count();
   }
   return i_ret;
 }
 
 int64_t ObMultiKmeansExecutor::get_centers_dim() const
 {
   int64_t i_ret = 0;
   if (!algos_.empty() && OB_NOT_NULL(algos_[0])) {
     i_ret = algos_[0]->get_cur_centers().dim_;
   }
   return i_ret;
 }
 
 int ObMultiKmeansExecutor::get_center(const int64_t pos, float *&center_vector)
 {
   int ret = OB_SUCCESS;
   int64_t centers_count = get_centers_count_per_kmeans();
   if (pos < 0 || pos >= get_total_centers_count()) {
     ret = OB_INDEX_OUT_OF_RANGE;
     LOG_WARN("index out of range", K(ret), K(pos), K(get_total_centers_count()));
   } else if (centers_count <= 0 || pos / centers_count >= algos_.count()) {
     ret = OB_INDEX_OUT_OF_RANGE;
     LOG_WARN("index out of range", K(ret), K(centers_count), K(pos), K(algos_.count()));
   } else {
     ObKmeansAlgo *algo = algos_[pos / centers_count];
     if (OB_ISNULL(algo)) {
       ret = OB_ERR_NULL_VALUE;
       LOG_WARN("invalid null algo", K(ret), K(pos / centers_count));
     } else if (pos % centers_count >= algo->get_cur_centers().count()) {
       ret = OB_INDEX_OUT_OF_RANGE;
       LOG_WARN("index out of range", K(ret), K(centers_count), K(pos), K(algo->get_cur_centers().count()));
     } else {
       center_vector = algo->get_cur_centers().at(pos % centers_count);
     }
   }
   return ret;
 }
 
 // ------------------ ObExternalGpuKmeansAlgo (out-of-process k-means) ------------------
 int ObExternalGpuKmeansAlgo::init_first_center(const ObIArray<float *> &input_vectors)
 {
   int ret = OB_SUCCESS;
   if (ob_external_gpu_kmeanspp_enabled()) {
     if (PREPARE_CENTERS != status_) {
       ret = OB_STATE_NOT_MATCH;
       SHARE_LOG(WARN, "status not match", K(ret), K(status_));
     } else {
       const int64_t center_init_cost_ms = ObTimeUtility::current_time_ms() - center_init_start_ms_;
       status_ = RUNNING_KMEANS;
       kmeans_log(
           "center_init_finished gpu_kmeanspp_skip_cpu center_init_cost_ms=%ld (default worker k-means++; set OB_EXTERNAL_GPU_KMEANSPP=0 for OB CPU k-means++)",
           center_init_cost_ms);
       SHARE_LOG(INFO, "external kmeans: skip CPU k-means++ init; worker runs k-means++ on GPU", K(ret));
     }
   } else {
     ret = ObKmeansAlgo::init_first_center(input_vectors);
   }
   return ret;
 }
 
 int ObExternalGpuKmeansAlgo::do_kmeans(const ObIArray<float *> &input_vectors)
 {
   int ret = OB_SUCCESS;
   char in_template[] = "/tmp/ob_ext_km_in_XXXXXX";
   char out_template[] = "/tmp/ob_ext_km_out_XXXXXX";
   int in_fd = ::mkstemp(in_template);
   int out_fd = ::mkstemp(out_template);
   std::vector<float> row_buf;
   if (RUNNING_KMEANS != status_) {
     ret = OB_STATE_NOT_MATCH;
     SHARE_LOG(WARN, "status not match", K(ret), K(status_));
     kmeans_log("external_kmeans_skip reason=bad_status status=%d", static_cast<int>(status_));
   } else if (OB_UNLIKELY(kmeans_ctx_->get_train_strategy() == KTS_NMBKM)) {
     ret = OB_NOT_SUPPORTED;
     SHARE_LOG(WARN, "external kmeans does not support NMBKM train strategy", K(ret));
     kmeans_log("external_kmeans_skip reason=nmbkm_not_supported");
   } else if (in_fd < 0 || out_fd < 0) {
     ret = OB_IO_ERROR;
     SHARE_LOG(WARN, "mkstemp failed", K(ret), K(in_fd), K(out_fd));
     kmeans_log("external_kmeans_fail reason=mkstemp in_fd=%d out_fd=%d", in_fd, out_fd);
   } else {
     ::close(out_fd);
     out_fd = -1;
     const char *cmd = ob_get_external_kmeans_cmd_or_default();
     const int max_iters = ob_external_kmeans_max_iters_from_env();
     const int64_t dim = kmeans_ctx_->dim_;
     const int64_t k = kmeans_ctx_->lists_;
     const bool gpu_kpp = ob_external_gpu_kmeanspp_enabled();
     const bool write_init = !gpu_kpp && (centers_[cur_idx_].count() == k);
     int64_t external_kmeans_wall_t0_ms = 0;
     const int32_t ctx_dist_algo = static_cast<int32_t>(kmeans_ctx_->dist_algo_);
     // PQ codebook: OB Elkan clusters raw subvectors with L2 (faiss-style); do not forward index-level COS.
     const int32_t worker_dist_algo = kmeans_ctx_->is_pq_stage_
         ? static_cast<int32_t>(VIDA_L2)
         : ctx_dist_algo;
     if (OB_ISNULL(cmd) || cmd[0] == '\0') {
       ret = OB_ERR_UNEXPECTED;
       SHARE_LOG(WARN, "external kmeans cmd empty (set OB_EXTERNAL_KMEANS_CMD or install $HOME/test/flash-kmeans + worker)",
           K(ret));
       kmeans_log("external_kmeans_fail reason=cmd_empty (set OB_EXTERNAL_KMEANS_CMD or default worker path)");
     } else if (input_vectors.count() <= 0 || dim <= 0 || k <= 0) {
       ret = OB_INVALID_ARGUMENT;
       SHARE_LOG(WARN, "invalid n/dim/k", K(ret), K(input_vectors.count()), K(dim), K(k));
       kmeans_log("external_kmeans_fail reason=invalid_n_dim_k n=%ld dim=%ld k=%ld",
           input_vectors.count(), dim, k);
     } else if (!gpu_kpp && !write_init) {
       ret = OB_ERR_UNEXPECTED;
       SHARE_LOG(WARN, "init centers count mismatch", K(ret), K(centers_[cur_idx_].count()), K(k));
       kmeans_log("external_kmeans_fail reason=init_centers_mismatch have=%ld need=%ld",
           centers_[cur_idx_].count(), k);
     } else {
       external_kmeans_wall_t0_ms = ObTimeUtility::current_time_ms();
       ::close(in_fd);
       in_fd = -1;
       row_buf.resize(static_cast<size_t>(dim));
       const int64_t write_input_t0_ms = ObTimeUtility::current_time_ms();
       if (kmeans_ctx_->is_pq_stage_ && ctx_dist_algo != worker_dist_algo) {
         kmeans_log(
             "external_kmeans: PQ stage worker dist_algo=%d (index dist_algo=%d -> L2 for faiss/Elkan parity)",
             worker_dist_algo,
             ctx_dist_algo);
       }
       if (OB_FAIL(write_ob_external_kmeans_input(
                      in_template,
                      input_vectors,
                      centers_[cur_idx_],
                      k,
                      dim,
                      worker_dist_algo,
                      max_iters,
                      write_init,
                      gpu_kpp))) {
         SHARE_LOG(WARN, "write external kmeans input failed", K(ret));
         kmeans_log("external_kmeans_fail reason=write_input_bin ret=%d", ret);
       } else {
         const int64_t write_input_ms = ObTimeUtility::current_time_ms() - write_input_t0_ms;
         kmeans_log(
             "external_kmeans_write_input_ms=%ld in=%s",
             write_input_ms,
             in_template);
         const int32_t dist_algo_i = worker_dist_algo;
         kmeans_log(
             "external_kmeans_invoke n=%ld k=%ld dim=%ld dist_algo=%d max_iters=%d write_init=%d gpu_kmeanspp=%d in=%s out=%s",
             input_vectors.count(),
             k,
             dim,
             dist_algo_i,
             max_iters,
             write_init ? 1 : 0,
             gpu_kpp ? 1 : 0,
             in_template,
             out_template);
         if (OB_ISNULL(::getenv("USE_FLASH_KMEANS")) && OB_NOT_NULL(std::strstr(cmd, "flash-kmeans"))) {
           (void)::setenv("USE_FLASH_KMEANS", "1", 0);
         }
         ob_external_kmeans_log_config(gpu_kpp, write_init, max_iters, cmd);
         char cmdline[2048];
         const int n = snprintf(cmdline,
             sizeof(cmdline),
             "%s \"%s\" \"%s\"",
             cmd,
             in_template,
             out_template);
         if (n <= 0 || n >= static_cast<int>(sizeof(cmdline))) {
           ret = OB_ERR_UNEXPECTED;
           SHARE_LOG(WARN, "kmeans cmd too long", K(ret), K(n));
           kmeans_log("external_kmeans_fail reason=cmdline_too_long n=%d", n);
         } else {
           const int64_t system_t0_ms = ObTimeUtility::current_time_ms();
           const int sys_ret = ::system(cmdline);
           const int64_t system_elapsed_ms = ObTimeUtility::current_time_ms() - system_t0_ms;
           kmeans_log(
               "external_kmeans_system_return sys_ret=%d system_elapsed_ms=%ld dist_algo=%d (worker exit; 0=success)",
               sys_ret,
               system_elapsed_ms,
               dist_algo_i);
           if (sys_ret != 0) {
             ret = OB_ERR_UNEXPECTED;
             SHARE_LOG(WARN, "external kmeans command failed", K(ret), K(sys_ret), K(cmdline));
             kmeans_log(
                 "external_kmeans_fail reason=worker_nonzero_exit sys_ret=%d system_elapsed_ms=%ld",
                 sys_ret,
                 system_elapsed_ms);
           }
         }
       }
     }
     if (in_fd >= 0) {
       ::close(in_fd);
       in_fd = -1;
     }
     (void)::unlink(in_template);
 
     if (OB_SUCC(ret)) {
       FILE *fp = fopen(out_template, "rb");
       if (OB_ISNULL(fp)) {
         ret = OB_IO_ERROR;
         SHARE_LOG(WARN, "fopen kmeans output failed", K(ret), KP(out_template));
         kmeans_log("external_kmeans_fail reason=fopen_output path=%s", out_template);
       } else {
         ObExtKmeansOutHeader hdr;
         if (fread(&hdr, sizeof(hdr), 1, fp) != 1) {
           ret = OB_IO_ERROR;
           SHARE_LOG(WARN, "fread output header failed", K(ret));
           kmeans_log("external_kmeans_fail reason=fread_output_header");
         } else if (hdr.magic_ != OB_EXT_KMEANS_MAGIC || hdr.version_ != OB_EXT_KMEANS_VERSION) {
           ret = OB_ERR_UNEXPECTED;
           SHARE_LOG(WARN, "bad output header", K(ret), K(hdr.magic_), K(hdr.version_));
           kmeans_log(
               "external_kmeans_fail reason=bad_output_magic_ver magic=%u ver=%u",
               hdr.magic_,
               hdr.version_);
         } else if (hdr.k_ != k || hdr.dim_ != dim) {
           ret = OB_ERR_UNEXPECTED;
           SHARE_LOG(WARN, "output k/dim mismatch", K(ret), K(hdr.k_), K(k), K(hdr.dim_), K(dim));
           kmeans_log(
               "external_kmeans_fail reason=output_k_dim_mismatch out_k=%ld expect_k=%ld out_dim=%ld expect_dim=%ld",
               hdr.k_,
               k,
               hdr.dim_,
               dim);
         } else if (OB_FAIL(centers_[next_idx()].init(dim, k, ivf_build_mem_ctx_))) {
           SHARE_LOG(WARN, "init centers buffer failed", K(ret));
           kmeans_log("external_kmeans_fail reason=init_centers_buffer ret=%d", ret);
         } else {
           for (int64_t i = 0; OB_SUCC(ret) && i < k; ++i) {
             if (fread(row_buf.data(), sizeof(float) * static_cast<size_t>(dim), 1, fp) != 1) {
               ret = OB_IO_ERROR;
               SHARE_LOG(WARN, "fread centroid row failed", K(ret), K(i));
               kmeans_log("external_kmeans_fail reason=fread_centroid_row i=%ld", i);
             } else if (OB_FAIL(centers_[next_idx()].push_back(dim, row_buf.data()))) {
               SHARE_LOG(WARN, "push_back center failed", K(ret), K(i));
               kmeans_log("external_kmeans_fail reason=push_back_center i=%ld ret=%d", i, ret);
             }
           }
         }
         (void)fclose(fp);
       }
       (void)::unlink(out_template);
     } else {
       (void)::unlink(out_template);
     }
 
     if (OB_SUCC(ret)) {
       for (int64_t i = 0; OB_SUCC(ret) && i < k; ++i) {
         if (OB_FAIL(kmeans_ctx_->try_normalize(dim, centers_[next_idx()].at(i), centers_[next_idx()].at(i)))) {
           LOG_WARN("failed to normalize external center", K(ret), K(i));
         }
       }
     }
     if (OB_SUCC(ret)) {
       cur_idx_ = next_idx();
       status_ = FINISH;
       SHARE_LOG(INFO, "external gpu kmeans finished", K(k), K(dim), K(input_vectors.count()), K(max_iters));
       const int64_t wall_ms =
           (external_kmeans_wall_t0_ms > 0) ? (ObTimeUtility::current_time_ms() - external_kmeans_wall_t0_ms) : -1;
       kmeans_log(
           "external_kmeans_done k=%ld dim=%ld n=%ld max_iters=%d dist_algo=%d wall_elapsed_ms=%ld (out-of-process worker)",
           k,
           dim,
           input_vectors.count(),
           max_iters,
           static_cast<int32_t>(worker_dist_algo),
           wall_ms);
     }
   }
   return ret;
 }
 
 // ------------------ ObElkanKmeansAlgo implement ------------------
 void ObElkanKmeansAlgo::destroy()
 {
   if (enable_hgraph_ && OB_NOT_NULL(hgraph_index_)) {
     obvectorutil::delete_index(hgraph_index_);
     hgraph_index_ = nullptr;
     LOG_DEBUG("Released HGraph index in destroy()");
   }
 
   ObKmeansAlgo::destroy();
 }
 
 int ObElkanKmeansAlgo::assign_vectors_parallel(const ObIArray<float *> &input_vectors, float *centers_distance,
                                                int32_t *data_cnt_in_cluster, float &dis_obj,
                                                int32_t *nearest_label_out, float *min_d2_out,
                                                const bool accumulate_to_centers,
                                                const bool allow_hgraph_assign)
 {
   int ret = OB_SUCCESS;
   const int64_t sample_cnt = input_vectors.count();
   if (sample_cnt <= 0) {
     dis_obj = 0.f;
     return ret;
   }
 
   if (OB_ISNULL(task_handler_)) {
     // If task handler is unavailable, fallback to serial processing
     if (OB_FAIL(assign_vectors_range(input_vectors, 0, sample_cnt, centers_distance, data_cnt_in_cluster, dis_obj,
                                      false, nearest_label_out, min_d2_out, accumulate_to_centers,
                                      allow_hgraph_assign))) {
       SHARE_LOG(WARN, "failed to assign vectors range", K(ret));
     }
   } else if (OB_UNLIKELY(OB_ISNULL(assign_tasks_) || max_assign_tasks_ <= 0)) {
     ret = OB_ERR_UNEXPECTED;
     SHARE_LOG(WARN, "unexpected nullptr", K(ret));
   } else { // parallel assign path
     dis_obj = 0.0f;
     // NOTE: max_assign_tasks_ + 1 is used to handle the final task
     const int64_t block_size = std::max(1L, sample_cnt / (max_assign_tasks_ + 1));
     int64_t end_idx = 0;
 
     // Create and submit tasks
     for (int64_t i = 0; OB_SUCC(ret) && i < max_assign_tasks_; ++i) {
       // max_assign_tasks_ is min(sample_cnt, max_thread_cnt), so end_idx <= sample_cnt
       int64_t start_idx = i * block_size;
       end_idx = start_idx + block_size;
 
       ObKmeansAssignTask &task = assign_tasks_[i];
       // Reset task state to ensure proper reuse
       task.reset();
       if (check_stop()) {
         ret = OB_CANCELED;
         SHARE_LOG(WARN, "check stop", K(ret));
       } else if (OB_FAIL(task.init(start_idx, end_idx, this, &input_vectors, centers_distance, data_cnt_in_cluster,
                                    nearest_label_out, min_d2_out, accumulate_to_centers, allow_hgraph_assign))) {
         SHARE_LOG(WARN, "failed to init assign task", K(ret), K(i));
       } else if (OB_FAIL(task_handler_->push_task(task))) {
         if (OB_EAGAIN != ret) {
           SHARE_LOG(WARN, "failed to push assign task", K(ret), K(i));
         } else if (OB_FAIL(task.do_work())) {
           SHARE_LOG(WARN, "failed to do assign task", K(ret));
         }
       }
     }
 
     // Handle the final task
     if (OB_SUCC(ret)) {
       float tmp_dis_obj = 0.0f;
       if (OB_FAIL(assign_vectors_range(input_vectors, end_idx, sample_cnt, centers_distance, data_cnt_in_cluster,
                                        tmp_dis_obj, true, nearest_label_out, min_d2_out, accumulate_to_centers,
                                        allow_hgraph_assign))) {
         SHARE_LOG(WARN, "failed to assign vectors range", K(ret), K(end_idx), K(sample_cnt));
       } else {
         dis_obj += tmp_dis_obj;
       }
     } else {
       for (int64_t i = 0; i < max_assign_tasks_; ++i) {
         ObKmeansAssignTask &task = assign_tasks_[i];
         task.set_task_stop();
       }
     }
 
     // Note. Anywhere above, all tasks need to be stopped otherwise it may cause a core dump because some data
     // will be released.
     wait_parallel_task_finish(assign_tasks_, max_assign_tasks_, *task_handler_);
 
     for (int64_t i = 0; OB_SUCC(ret) && i < max_assign_tasks_; ++i) {
       ObKmeansAssignTask &task = assign_tasks_[i];
       if (OB_FAIL(task.get_ret())) {
         SHARE_LOG(WARN, "assign task failed", K(ret), K(i));
       } else {
         dis_obj += task.get_dis_obj();
       }
     }
   }
 
   return ret;
 }
 
 int ObElkanKmeansAlgo::search_nearest_center(const ObIArray<float*> &input_vectors, float* centers_distance, int32_t *data_cnt_in_cluster, float &dis_obj,
                                              int32_t *nearest_label_out)
 {
   int ret = OB_SUCCESS;
   const bool use_hgraph = is_hgraph_available();
   if (OB_UNLIKELY(kmeans_ctx_->lists_ != centers_[cur_idx_].count() || OB_ISNULL(data_cnt_in_cluster))) {
     ret = OB_ERR_UNEXPECTED;
     SHARE_LOG(WARN, "param error", K(ret), K(kmeans_ctx_->lists_), K(centers_[cur_idx_].count()),
         KP(centers_distance), KP(data_cnt_in_cluster), K(use_hgraph));
   } else if (use_hgraph) {
     if (OB_UNLIKELY(nullptr == hgraph_index_)) {
       ret = OB_ERR_UNEXPECTED;
       SHARE_LOG(WARN, "hgraph enabled but index not ready", K(ret));
     } else if (OB_FAIL(assign_vectors_parallel(
                    input_vectors, nullptr, data_cnt_in_cluster, dis_obj, nearest_label_out, nullptr, true, true))) {
       SHARE_LOG(WARN, "failed to assign vectors with hgraph", K(ret));
     } else {
       dis_obj = dis_obj / std::max(1L, input_vectors.count());
     }
   } else if (OB_ISNULL(centers_distance)) {
     ret = OB_INVALID_ARGUMENT;
     SHARE_LOG(WARN, "centers_distance is null for brute-force search", K(ret));
   } else {
     const int64_t sample_cnt = input_vectors.count();
     const int64_t center_count = kmeans_ctx_->lists_;
     const int64_t dim = kmeans_ctx_->dim_;
 
     float distance = 0.0;
     // 1. calc distance between each two centers
     for (int64_t i = 0; OB_SUCC(ret) && i < center_count; ++i) {
       for (int64_t j = i + 1; OB_SUCC(ret) && j < center_count; ++j) {
         if (OB_FAIL(calc_kmeans_distance(centers_[cur_idx_].at(i), centers_[cur_idx_].at(j), dim, distance))) {
           SHARE_LOG(WARN, "failed to calc kmeans distance between centers", K(ret));
         } else {
           set_centers_distance(centers_distance, i, j, distance);
         }
       }
     }
     if (OB_SUCC(ret)) {
       // Use block parallel processing for vector assignment
       if (OB_FAIL(assign_vectors_parallel(
               input_vectors, centers_distance, data_cnt_in_cluster, dis_obj, nearest_label_out, nullptr, true, true))) {
         SHARE_LOG(WARN, "failed to assign vectors parallel", K(ret));
       } else {
         dis_obj = dis_obj / sample_cnt;
       }
     }
   }
   return ret;
 }
 
 int ObElkanKmeansAlgo::build_hgraph_for_centers()
 {
   int ret = OB_SUCCESS;
 
   if (OB_ISNULL(kmeans_ctx_)) {
     ret = OB_ERR_NULL_VALUE;
     LOG_WARN("kmeans_ctx is null", K(ret));
   } else {
     const int64_t nlist = kmeans_ctx_->lists_;
     const int64_t dim = kmeans_ctx_->dim_;
     ObVectorIndexParam hgraph_param;
     hgraph_param.nlist_ = nlist;
     hgraph_param.dim_ = dim;
     hgraph_param.dist_algorithm_ = ObVectorIndexDistAlgorithm::VIDA_L2;
     hgraph_param.m_ = 16;
     hgraph_param.ef_construction_ = 200;
     hgraph_param.ef_search_ = 64;
     hgraph_param.extra_info_actual_size_ = 0;
     hgraph_param.refine_type_ = 0;
     hgraph_param.bq_bits_query_ = ObVectorIndexParam::DEFAULT_BQ_BITS_QUERY;
     hgraph_param.bq_use_fht_ = false;
 
     lib::ObMallocHookAttrGuard malloc_guard(lib::ObMemAttr(kmeans_ctx_->tenant_id_, "KmeansHGraphEst"));
 
     // 1. estimate memory for hgraph construction
     int64_t estimated_memory = 0;
     if (OB_FAIL(ObVectorIndexUtil::estimate_hgraph_memory_for_ivf_centers(
             hgraph_param, estimated_memory))) {
       LOG_WARN("failed to estimate hgraph memory", K(ret), K(nlist), K(dim));
     } else {
       estimated_memory += nlist * sizeof(int64_t);
       // 2. check memory limit
       int64_t current_memory = ivf_build_mem_ctx_.get_all_vsag_use_mem_byte();
       int64_t memory_limit = 0;
       LOG_INFO("HGraph build memory check", K(current_memory), K(estimated_memory), K(nlist), K(dim));
 
       if (OB_FAIL(ObPluginVectorIndexHelper::get_vector_memory_limit_size(
               kmeans_ctx_->tenant_id_, memory_limit))) {
         LOG_WARN("failed to get memory limit", K(ret), K(kmeans_ctx_->tenant_id_));
       } else if (current_memory + estimated_memory > memory_limit) {
         ret = OB_ALLOCATE_MEMORY_FAILED;
         LOG_WARN("insufficient memory for HGraph construction",
                  K(ret), K(current_memory), K(estimated_memory), K(memory_limit), K(nlist), K(dim));
       } else {
         // 3. memory enough, build hgraph
         LOG_INFO("Starting HGraph construction with memory check",
                  K(nlist), K(dim), K(estimated_memory), K(current_memory), K(memory_limit));
         ret = do_build_hgraph_for_centers(hgraph_param);
       }
     }
   }
 
   return ret;
 }
 
 int ObElkanKmeansAlgo::do_build_hgraph_for_centers(const ObVectorIndexParam &param)
 {
   int ret = OB_SUCCESS;
   const int64_t nlist = param.nlist_;
   const int64_t dim = param.dim_;
 
   if (OB_NOT_NULL(hgraph_index_)) {
     obvectorutil::delete_index(hgraph_index_);
     hgraph_index_ = nullptr;
     LOG_DEBUG("Released old HGraph index before rebuilding", K(nlist), K(dim));
   }
 
   ObCentersBuffer<float> &centers = get_cur_centers();
   if (centers.count() != nlist) {
     ret = OB_ERR_UNEXPECTED;
     LOG_WARN("centers buffer not ready", K(ret), K(centers.count()), K(nlist));
   } else {
     // Initialize hgraph_allocator_ if not already initialized
     if (!hgraph_allocator_.is_inited()) {
       if (OB_ISNULL(kmeans_ctx_)) {
         ret = OB_ERR_UNEXPECTED;
         LOG_WARN("kmeans_ctx_ is null, cannot init hgraph allocator", K(ret));
       } else if (OB_FAIL(hgraph_allocator_.init(
           ivf_build_mem_ctx_.get_mem_context(),
           ivf_build_mem_ctx_.get_all_vsag_use_mem(),
           kmeans_ctx_->tenant_id_))) {
         LOG_WARN("failed to init hgraph allocator", K(ret), K(kmeans_ctx_->tenant_id_));
       }
     }
 
     if (OB_FAIL(ret)) {
       // initialization failed, skip hgraph build
     } else {
       const char *metric = "l2";
       int64_t memory_before = ivf_build_mem_ctx_.get_all_vsag_use_mem_byte();
 
       if (OB_FAIL(obvectorutil::create_index(
         hgraph_index_,
         VIAT_HGRAPH,
         "float32",
         metric,
         dim,
         param.m_,
         param.ef_construction_,
         param.ef_search_,
         &hgraph_allocator_,
         param.extra_info_actual_size_,
         param.refine_type_,
         param.bq_bits_query_,
         param.bq_use_fht_))) {
         LOG_WARN("fail to create hgraph index", K(ret), K(nlist), K(dim));
       } else {
         LOG_DEBUG("Successfully created HGraph index, now adding centers data", K(nlist), K(dim));
 
         float *all_centers_data = centers.at(0);
         int64_t *center_ids = static_cast<int64_t *>(ivf_build_mem_ctx_.Allocate(sizeof(int64_t) * nlist));
         if (OB_ISNULL(all_centers_data) || OB_ISNULL(center_ids)) {
           ret = OB_ALLOCATE_MEMORY_FAILED;
           LOG_WARN("failed to get centers data or allocate ids", K(ret), KP(all_centers_data), KP(center_ids));
         } else {
           // prepare id array
           for (int64_t i = 0; i < nlist; ++i) {
             center_ids[i] = i + 1; // id starts from 1
           }
           if (OB_FAIL(obvectorutil::build_index(hgraph_index_, all_centers_data, center_ids, dim, nlist))) {
             LOG_WARN("failed to build hgraph index", K(ret), K(nlist), K(dim));
             obvectorutil::delete_index(hgraph_index_);
             hgraph_index_ = nullptr;
           } else {
             int64_t memory_after = ivf_build_mem_ctx_.get_all_vsag_use_mem_byte();
             int64_t actual_used = memory_after - memory_before;
             LOG_INFO("HGraph built successfully for K-means",
                     K(nlist), K(dim), K(memory_before), K(memory_after), K(actual_used));
           }
           // clean id array
           if (center_ids) {
             ivf_build_mem_ctx_.Deallocate(center_ids);
           }
         }
       }
     }
   }
 
   return ret;
 }
 
 namespace {
 OB_INLINE void nmbkm_row_add(float *row, const float *x, const int64_t dim)
 {
   for (int64_t d = 0; d < dim; ++d) {
     row[d] += x[d];
   }
 }
 OB_INLINE float *nmbkm_cluster_row(float *S_sum, const int64_t dim, const int64_t j)
 {
   return S_sum + j * dim;
 }
 // Prefix batch growth: while b < n/2 use doubling (cap n); once b >= n/2 add max(floor((n-b)/2),1) per step.
 OB_INLINE int64_t nmbkm_next_batch_size(const int64_t b, const int64_t n)
 {
   if (b >= n) {
     return n;
   }
   if (b >= n / 2) {
     const int64_t left = n - b;
     const int64_t step = std::max(left / 2, static_cast<int64_t>(1));
     return std::min(b + step, n);
   }
   return std::min(b * 2, n);
 }
 // Prefix batch stats -> next center: cluster sum S_row with count vj, else keep old_center.
 OB_INLINE void nmbkm_fill_center_mean_or_old(
     float *dst,
     const float *S_row,
     const int64_t dim,
     const int32_t vj,
     const float *old_center)
 {
   if (vj > 0) {
     const float inv_vj = 1.f / static_cast<float>(vj);
     for (int64_t d = 0; d < dim; ++d) {
       dst[d] = S_row[d] * inv_vj;
     }
   } else {
     MEMCPY(dst, old_center, sizeof(float) * static_cast<size_t>(dim));
   }
 }
 
 static float nmbkm_dot_f(const float *a, const float *b, const int64_t dim)
 {
   float s = 0.f;
   for (int64_t d = 0; d < dim; ++d) {
     s += a[d] * b[d];
   }
   return s;
 }
 
 static void nmbkm_axpy_inplace_minus_mu(float *dst, const float *x, const float *mu, const int64_t dim)
 {
   for (int64_t d = 0; d < dim; ++d) {
     dst[d] = x[d] - mu[d];
   }
 }
 
 // First principal direction via power iteration on covariance; median split -> two means (not necessarily unit).
 static int nmbkm_pca_median_split_two_means(
     ObIvfMemContext &mem,
     const ObIArray<float *> &input_vectors,
     const int64_t *indices,
     const int64_t n_local,
     const int64_t dim,
     float *out_a,
     float *out_b)
 {
   int ret = OB_SUCCESS;
   float *mu = nullptr;
   float *v = nullptr;
   float *w = nullptr;
   float *xc = nullptr;
   float *proj = nullptr;
   float *proj_copy = nullptr;
   if (n_local < 2 || OB_ISNULL(indices) || OB_ISNULL(out_a) || OB_ISNULL(out_b)) {
     ret = OB_INVALID_ARGUMENT;
   } else if (OB_ISNULL(mu = static_cast<float *>(mem.Allocate(sizeof(float) * static_cast<size_t>(dim))))) {
     ret = OB_ALLOCATE_MEMORY_FAILED;
   } else if (OB_ISNULL(v = static_cast<float *>(mem.Allocate(sizeof(float) * static_cast<size_t>(dim))))) {
     ret = OB_ALLOCATE_MEMORY_FAILED;
   } else if (OB_ISNULL(w = static_cast<float *>(mem.Allocate(sizeof(float) * static_cast<size_t>(dim))))) {
     ret = OB_ALLOCATE_MEMORY_FAILED;
   } else if (OB_ISNULL(xc = static_cast<float *>(mem.Allocate(sizeof(float) * static_cast<size_t>(dim))))) {
     ret = OB_ALLOCATE_MEMORY_FAILED;
   } else if (OB_ISNULL(proj = static_cast<float *>(mem.Allocate(sizeof(float) * static_cast<size_t>(n_local))))) {
     ret = OB_ALLOCATE_MEMORY_FAILED;
   } else if (OB_ISNULL(proj_copy = static_cast<float *>(mem.Allocate(sizeof(float) * static_cast<size_t>(n_local))))) {
     ret = OB_ALLOCATE_MEMORY_FAILED;
   } else {
     MEMSET(mu, 0, sizeof(float) * static_cast<size_t>(dim));
     for (int64_t i = 0; i < n_local; ++i) {
       const float *xi = input_vectors.at(indices[i]);
       for (int64_t d = 0; d < dim; ++d) {
         mu[d] += xi[d];
       }
     }
     const float inv_n = 1.f / static_cast<float>(n_local);
     for (int64_t d = 0; d < dim; ++d) {
       mu[d] *= inv_n;
     }
     if (n_local == 2) {
       MEMCPY(out_a, input_vectors.at(indices[0]), sizeof(float) * static_cast<size_t>(dim));
       MEMCPY(out_b, input_vectors.at(indices[1]), sizeof(float) * static_cast<size_t>(dim));
     } else {
       {
         const float *x0 = input_vectors.at(indices[0]);
         for (int64_t d = 0; d < dim; ++d) {
           v[d] = x0[d] - mu[d];
         }
         float nv = std::sqrt(std::max(0.f, nmbkm_dot_f(v, v, dim)));
         if (nv < 1e-20f) {
           const float *x1 = input_vectors.at(indices[1]);
           for (int64_t d = 0; d < dim; ++d) {
             v[d] = x1[d] - mu[d];
           }
           nv = std::sqrt(std::max(0.f, nmbkm_dot_f(v, v, dim)));
         }
         if (nv < 1e-20f) {
           for (int64_t d = 0; d < dim; ++d) {
             v[d] = (d == 0) ? 1.f : 0.f;
           }
           nv = 1.f;
         } else {
           const float inv = 1.f / nv;
           for (int64_t d = 0; d < dim; ++d) {
             v[d] *= inv;
           }
         }
       }
       for (int it = 0; it < NMBKM_PCA_POWER_ITERS; ++it) {
         MEMSET(w, 0, sizeof(float) * static_cast<size_t>(dim));
         for (int64_t i = 0; i < n_local; ++i) {
           const float *xi = input_vectors.at(indices[i]);
           nmbkm_axpy_inplace_minus_mu(xc, xi, mu, dim);
           const float coeff = nmbkm_dot_f(xc, v, dim);
           for (int64_t d = 0; d < dim; ++d) {
             w[d] += coeff * xc[d];
           }
         }
         float nw = std::sqrt(std::max(0.f, nmbkm_dot_f(w, w, dim)));
         if (nw < 1e-20f) {
           break;
         }
         const float invw = 1.f / nw;
         for (int64_t d = 0; d < dim; ++d) {
           v[d] = w[d] * invw;
         }
       }
       for (int64_t i = 0; i < n_local; ++i) {
         const float *xi = input_vectors.at(indices[i]);
         nmbkm_axpy_inplace_minus_mu(xc, xi, mu, dim);
         proj[i] = nmbkm_dot_f(xc, v, dim);
       }
       MEMCPY(proj_copy, proj, sizeof(float) * static_cast<size_t>(n_local));
       const int64_t mid = n_local / 2;
       std::nth_element(proj_copy, proj_copy + mid, proj_copy + n_local);
       const float thr = proj_copy[mid];
       MEMSET(out_a, 0, sizeof(float) * static_cast<size_t>(dim));
       MEMSET(out_b, 0, sizeof(float) * static_cast<size_t>(dim));
       int64_t ca = 0;
       int64_t cb = 0;
       for (int64_t i = 0; i < n_local; ++i) {
         const float *xi = input_vectors.at(indices[i]);
         if (proj[i] <= thr) {
           for (int64_t d = 0; d < dim; ++d) {
             out_a[d] += xi[d];
           }
           ++ca;
         } else {
           for (int64_t d = 0; d < dim; ++d) {
             out_b[d] += xi[d];
           }
           ++cb;
         }
       }
       if (ca == 0 || cb == 0) {
         const int64_t split = n_local / 2;
         MEMSET(out_a, 0, sizeof(float) * static_cast<size_t>(dim));
         MEMSET(out_b, 0, sizeof(float) * static_cast<size_t>(dim));
         for (int64_t i = 0; i < split; ++i) {
           const float *xi = input_vectors.at(indices[i]);
           for (int64_t d = 0; d < dim; ++d) {
             out_a[d] += xi[d];
           }
         }
         for (int64_t i = split; i < n_local; ++i) {
           const float *xi = input_vectors.at(indices[i]);
           for (int64_t d = 0; d < dim; ++d) {
             out_b[d] += xi[d];
           }
         }
         ca = split;
         cb = n_local - split;
       }
       if (ca > 0) {
         const float ia = 1.f / static_cast<float>(ca);
         for (int64_t d = 0; d < dim; ++d) {
           out_a[d] *= ia;
         }
       }
       if (cb > 0) {
         const float ib = 1.f / static_cast<float>(cb);
         for (int64_t d = 0; d < dim; ++d) {
           out_b[d] *= ib;
         }
       }
     }
   }
 #define NMBKM_PCA_TMP_FREE(p)           \
   do {                                  \
     if (OB_NOT_NULL(p)) {               \
       mem.Deallocate(p);                \
       (p) = nullptr;                   \
     }                                  \
   } while (0)
   NMBKM_PCA_TMP_FREE(mu);
   NMBKM_PCA_TMP_FREE(v);
   NMBKM_PCA_TMP_FREE(w);
   NMBKM_PCA_TMP_FREE(xc);
   NMBKM_PCA_TMP_FREE(proj);
   NMBKM_PCA_TMP_FREE(proj_copy);
 #undef NMBKM_PCA_TMP_FREE
   return ret;
 }
 
 // One PC1; sort by projection; partition into m equal-frequency segments; m means (m>=3).
 static int nmbkm_pca_multi_split_m_means(ObIvfMemContext &mem, const ObIArray<float *> &input_vectors,
     const int64_t *indices, const int64_t n_local, const int64_t dim, const int64_t m, float **dst_rows)
 {
   int ret = OB_SUCCESS;
   if (m < 3 || n_local < m || OB_ISNULL(indices) || OB_ISNULL(dst_rows)) {
     ret = OB_INVALID_ARGUMENT;
   } else {
     for (int64_t r = 0; OB_SUCC(ret) && r < m; ++r) {
       if (OB_ISNULL(dst_rows[r])) {
         ret = OB_INVALID_ARGUMENT;
       }
     }
   }
   float *mu = nullptr;
   float *v = nullptr;
   float *w = nullptr;
   float *xc = nullptr;
   float *proj = nullptr;
   if (OB_SUCC(ret) && OB_ISNULL(mu = static_cast<float *>(mem.Allocate(sizeof(float) * static_cast<size_t>(dim))))) {
     ret = OB_ALLOCATE_MEMORY_FAILED;
   } else if (OB_SUCC(ret) && OB_ISNULL(v = static_cast<float *>(mem.Allocate(sizeof(float) * static_cast<size_t>(dim))))) {
     ret = OB_ALLOCATE_MEMORY_FAILED;
   } else if (OB_SUCC(ret) && OB_ISNULL(w = static_cast<float *>(mem.Allocate(sizeof(float) * static_cast<size_t>(dim))))) {
     ret = OB_ALLOCATE_MEMORY_FAILED;
   } else if (OB_SUCC(ret) && OB_ISNULL(xc = static_cast<float *>(mem.Allocate(sizeof(float) * static_cast<size_t>(dim))))) {
     ret = OB_ALLOCATE_MEMORY_FAILED;
   } else if (OB_SUCC(ret) && OB_ISNULL(proj = static_cast<float *>(mem.Allocate(sizeof(float) * static_cast<size_t>(n_local))))) {
     ret = OB_ALLOCATE_MEMORY_FAILED;
   } else if (OB_SUCC(ret)) {
     MEMSET(mu, 0, sizeof(float) * static_cast<size_t>(dim));
     for (int64_t i = 0; i < n_local; ++i) {
       const float *xi = input_vectors.at(indices[i]);
       for (int64_t d = 0; d < dim; ++d) {
         mu[d] += xi[d];
       }
     }
     const float inv_n = 1.f / static_cast<float>(n_local);
     for (int64_t d = 0; d < dim; ++d) {
       mu[d] *= inv_n;
     }
     {
       const float *x0 = input_vectors.at(indices[0]);
       for (int64_t d = 0; d < dim; ++d) {
         v[d] = x0[d] - mu[d];
       }
       float nv = std::sqrt(std::max(0.f, nmbkm_dot_f(v, v, dim)));
       if (nv < 1e-20f && n_local > 1) {
         const float *x1 = input_vectors.at(indices[1]);
         for (int64_t d = 0; d < dim; ++d) {
           v[d] = x1[d] - mu[d];
         }
         nv = std::sqrt(std::max(0.f, nmbkm_dot_f(v, v, dim)));
       }
       if (nv < 1e-20f) {
         for (int64_t d = 0; d < dim; ++d) {
           v[d] = (d == 0) ? 1.f : 0.f;
         }
         nv = 1.f;
       } else {
         const float inv = 1.f / nv;
         for (int64_t d = 0; d < dim; ++d) {
           v[d] *= inv;
         }
       }
     }
     for (int it = 0; it < NMBKM_PCA_POWER_ITERS; ++it) {
       MEMSET(w, 0, sizeof(float) * static_cast<size_t>(dim));
       for (int64_t i = 0; i < n_local; ++i) {
         const float *xi = input_vectors.at(indices[i]);
         nmbkm_axpy_inplace_minus_mu(xc, xi, mu, dim);
         const float coeff = nmbkm_dot_f(xc, v, dim);
         for (int64_t d = 0; d < dim; ++d) {
           w[d] += coeff * xc[d];
         }
       }
       float nw = std::sqrt(std::max(0.f, nmbkm_dot_f(w, w, dim)));
       if (nw < 1e-20f) {
         break;
       }
       const float invw = 1.f / nw;
       for (int64_t d = 0; d < dim; ++d) {
         v[d] = w[d] * invw;
       }
     }
     for (int64_t i = 0; i < n_local; ++i) {
       const float *xi = input_vectors.at(indices[i]);
       nmbkm_axpy_inplace_minus_mu(xc, xi, mu, dim);
       proj[i] = nmbkm_dot_f(xc, v, dim);
     }
     std::vector<int64_t> ord(static_cast<size_t>(n_local));
     std::iota(ord.begin(), ord.end(), static_cast<int64_t>(0));
     std::sort(ord.begin(), ord.end(),
         [&](const int64_t a, const int64_t b) {
           return proj[a] < proj[b];
         });
     int64_t off = 0;
     for (int64_t r = 0; OB_SUCC(ret) && r < m; ++r) {
       const int64_t seg = n_local / m + (r < (n_local % m) ? 1 : 0);
       MEMSET(dst_rows[r], 0, sizeof(float) * static_cast<size_t>(dim));
       for (int64_t t = 0; t < seg; ++t) {
         const int64_t local_ix = ord[static_cast<size_t>(off + t)];
         const float *xi = input_vectors.at(indices[local_ix]);
         for (int64_t d = 0; d < dim; ++d) {
           dst_rows[r][d] += xi[d];
         }
       }
       if (seg > 0) {
         const float invs = 1.f / static_cast<float>(seg);
         for (int64_t d = 0; d < dim; ++d) {
           dst_rows[r][d] *= invs;
         }
       }
       off += seg;
     }
   }
 #define NMBKM_PCA_M_FREE(p)             \
   do {                                  \
     if (OB_NOT_NULL(p)) {               \
       mem.Deallocate(p);                \
       (p) = nullptr;                   \
     }                                  \
   } while (0)
   NMBKM_PCA_M_FREE(mu);
   NMBKM_PCA_M_FREE(v);
   NMBKM_PCA_M_FREE(w);
   NMBKM_PCA_M_FREE(xc);
   NMBKM_PCA_M_FREE(proj);
 #undef NMBKM_PCA_M_FREE
   return ret;
 }
 }  // namespace
 
 int ObElkanKmeansAlgo::nmbkm_postprocess_pca_fill_empty_centers(const ObIArray<float *> &input_vectors,
     int32_t *data_cnt_in_cluster, float *centers_distance, int32_t *nearest_labels_opt,
     int64_t nmbkm_batch_for_sparse_thr)
 {
   if (!NMBKM_POST_PCA_FILL_EMPTY_ENABLED) {
     return OB_SUCCESS;
   }
   int ret = OB_SUCCESS;
   if (OB_ISNULL(kmeans_ctx_) || OB_ISNULL(data_cnt_in_cluster)) {
     ret = OB_INVALID_ARGUMENT;
     SHARE_LOG(WARN, "nmbkm postprocess: invalid args", K(ret), KP(kmeans_ctx_), KP(data_cnt_in_cluster));
     return ret;
   }
   const int64_t n = input_vectors.count();
   const int64_t k = kmeans_ctx_->lists_;
   const int64_t dim = kmeans_ctx_->dim_;
   if (n <= 0 || k <= 0 || dim <= 0) {
     ret = OB_INVALID_ARGUMENT;
     SHARE_LOG(WARN, "nmbkm postprocess: bad n/k/dim", K(ret), K(n), K(k), K(dim));
     return ret;
   }
   if (centers_[cur_idx_].count() != k) {
     ret = OB_ERR_UNEXPECTED;
     SHARE_LOG(WARN, "nmbkm postprocess: center count mismatch", K(ret), K(k), K(centers_[cur_idx_].count()));
     return ret;
   }
 
   const int64_t batch_sz =
       (nmbkm_batch_for_sparse_thr > 0) ? nmbkm_batch_for_sparse_thr : n;
   const int64_t sparse_from_batch =
       batch_sz / std::max(static_cast<int64_t>(1), k * NMBKM_POST_PCA_SPARSE_DIV);
   const int64_t sparse_thr = std::max(static_cast<int64_t>(1),
       std::min(static_cast<int64_t>(NMBKM_POST_PCA_SPARSE_ABS_MAX), sparse_from_batch));
   const int64_t donor_min_cnt = std::max(static_cast<int64_t>(2), sparse_thr);
   const auto has_sparse_list_by_count = [&]() -> bool {
     for (int64_t j = 0; j < k; ++j) {
       if (data_cnt_in_cluster[j] < sparse_thr) {
         return true;
       }
     }
     return false;
   };
 
   int32_t *labels = nearest_labels_opt;
   bool labels_owned = false;
   if (OB_ISNULL(labels)) {
     if (OB_ISNULL(labels = static_cast<int32_t *>(ivf_build_mem_ctx_.Allocate(sizeof(int32_t) * static_cast<size_t>(n))))) {
       ret = OB_ALLOCATE_MEMORY_FAILED;
       SHARE_LOG(WARN, "nmbkm postprocess: alloc labels failed", K(ret), K(n));
       return ret;
     }
     labels_owned = true;
     float dis_lbl = 0.f;
     if (OB_FAIL(assign_vectors_parallel(
             input_vectors, centers_distance, nullptr, dis_lbl, labels, nullptr, false, true))) {
       SHARE_LOG(WARN, "nmbkm postprocess: label assign failed", K(ret));
       ivf_build_mem_ctx_.Deallocate(labels);
       return ret;
     }
     MEMSET(data_cnt_in_cluster, 0, sizeof(int32_t) * static_cast<size_t>(k));
     for (int64_t i = 0; i < n; ++i) {
       const int32_t lj = labels[i];
       if (OB_LIKELY(lj >= 0 && lj < k)) {
         ++data_cnt_in_cluster[lj];
       }
     }
     if (!has_sparse_list_by_count()) {
       kmeans_log(
           "nmbkm_post_pca: skip (all lists >= sparse_thr=%ld after full assign histogram)", sparse_thr);
       ivf_build_mem_ctx_.Deallocate(labels);
       return OB_SUCCESS;
     }
   } else if (!has_sparse_list_by_count()) {
     kmeans_log("nmbkm_post_pca: skip (all lists >= sparse_thr=%ld by count)", sparse_thr);
     return OB_SUCCESS;
   }
 
   std::vector<std::vector<int64_t>> cluster_pts(static_cast<size_t>(k));
   std::vector<int64_t> free_slots;
   free_slots.reserve(static_cast<size_t>(k));
   std::vector<std::pair<int32_t, int64_t>> donors;
   donors.reserve(static_cast<size_t>(k));
   for (int64_t j = 0; j < k; ++j) {
     const int32_t cj = data_cnt_in_cluster[j];
     cluster_pts[static_cast<size_t>(j)].reserve(static_cast<size_t>(std::max(0, cj)));
     const int64_t sz = static_cast<int64_t>(cj);
     if (sz < sparse_thr) {
       free_slots.push_back(j);
     } else if (sz >= donor_min_cnt) {
       donors.emplace_back(static_cast<int32_t>(sz), j);
     }
   }
   std::sort(donors.begin(), donors.end(), [](const std::pair<int32_t, int64_t> &a,
                                               const std::pair<int32_t, int64_t> &b) {
     return a.first != b.first ? a.first > b.first : a.second < b.second;
   });
   const int64_t n_free0 = static_cast<int64_t>(free_slots.size());
   if (free_slots.empty()) {
     kmeans_log("nmbkm_post_pca: no free slot (label size < sparse_thr=%ld), skip", sparse_thr);
     if (labels_owned) {
       ivf_build_mem_ctx_.Deallocate(labels);
     }
     return OB_SUCCESS;
   }
   if (donors.empty()) {
     kmeans_log("nmbkm_post_pca: free=%ld but no donor (size>=%ld), skip", n_free0, donor_min_cnt);
     if (labels_owned) {
       ivf_build_mem_ctx_.Deallocate(labels);
     }
     return OB_SUCCESS;
   }

   for (int64_t i = 0; i < n; ++i) {
     const int32_t lj = labels[i];
     if (OB_LIKELY(lj >= 0 && lj < k)) {
       cluster_pts[static_cast<size_t>(lj)].push_back(i);
     }
   }
 
   std::vector<int64_t> remaining_free = std::move(free_slots);
   const double mean_per_center = static_cast<double>(n) / static_cast<double>(k);
   int64_t slots_filled = 0;
   int64_t donors_processed = 0;
 
   for (size_t di = 0; OB_SUCC(ret) && di < donors.size() && !remaining_free.empty(); ++di) {
     const int64_t donor_j = donors[di].second;
     const std::vector<int64_t> &idxs = cluster_pts[static_cast<size_t>(donor_j)];
     const int64_t n_local = static_cast<int64_t>(idxs.size());
     const int64_t e_rem = static_cast<int64_t>(remaining_free.size());
     int64_t m_cap = (n_local * k > n) ? (n_local * k - 1) / n : 1;
     m_cap = std::min<int64_t>(m_cap, e_rem + 1);
     m_cap = std::min<int64_t>(m_cap, n_local);
     int64_t m_use = 2;
     if (m_cap >= 3 && static_cast<double>(n_local) / static_cast<double>(m_cap) > mean_per_center) {
       m_use = m_cap;
     }
     if (m_use >= 3) {
       std::vector<int64_t> popped;
       popped.reserve(static_cast<size_t>(m_use - 1));
       for (int64_t u = 0; u < m_use - 1; ++u) {
         popped.push_back(remaining_free.back());
         remaining_free.pop_back();
       }
       std::vector<float *> dst_rows(static_cast<size_t>(m_use));
       dst_rows[0] = centers_[cur_idx_].at(donor_j);
       for (int64_t u = 1; u < m_use; ++u) {
         dst_rows[static_cast<size_t>(u)] = centers_[cur_idx_].at(popped[static_cast<size_t>(u - 1)]);
       }
       if (OB_FAIL(nmbkm_pca_multi_split_m_means(
               ivf_build_mem_ctx_, input_vectors, idxs.data(), n_local, dim, m_use, dst_rows.data()))) {
         for (auto it = popped.rbegin(); it != popped.rend(); ++it) {
           remaining_free.push_back(*it);
         }
         SHARE_LOG(WARN, "nmbkm postprocess: multi PCA split failed", K(ret), K(donor_j), K(m_use));
       } else {
         bool norm_ok = true;
         for (int64_t u = 0; norm_ok && u < m_use; ++u) {
           if (OB_FAIL(kmeans_ctx_->try_normalize(dim, dst_rows[static_cast<size_t>(u)], dst_rows[static_cast<size_t>(u)]))) {
             SHARE_LOG(WARN, "nmbkm postprocess: normalize failed after multi split", K(ret), K(donor_j), K(u));
             norm_ok = false;
           }
         }
         if (norm_ok) {
           slots_filled += m_use - 1;
           ++donors_processed;
         }
       }
     } else {
       const int64_t slot_j = remaining_free.back();
       remaining_free.pop_back();
       float *dst_donor = centers_[cur_idx_].at(donor_j);
       float *dst_b = centers_[cur_idx_].at(slot_j);
       if (OB_FAIL(nmbkm_pca_median_split_two_means(
               ivf_build_mem_ctx_, input_vectors, idxs.data(), n_local, dim, dst_donor, dst_b))) {
         SHARE_LOG(WARN, "nmbkm postprocess: PCA split failed", K(ret), K(donor_j), K(slot_j));
         remaining_free.push_back(slot_j);
       } else if (OB_FAIL(kmeans_ctx_->try_normalize(dim, dst_donor, dst_donor))) {
         SHARE_LOG(WARN, "nmbkm postprocess: normalize donor failed", K(ret), K(donor_j));
         remaining_free.push_back(slot_j);
       } else if (OB_FAIL(kmeans_ctx_->try_normalize(dim, dst_b, dst_b))) {
         SHARE_LOG(WARN, "nmbkm postprocess: normalize free-slot center failed", K(ret), K(slot_j));
         remaining_free.push_back(slot_j);
       } else {
         ++slots_filled;
         ++donors_processed;
       }
     }
   }

   if (donors_processed == 0 && !remaining_free.empty()) {
     LOG_WARN("nmbkm postprocess: no split succeeded", K(remaining_free.size()));
   }
   kmeans_log(
       "nmbkm_post_pca: batch_sz=%ld sparse_from_batch=%ld abs_max=%ld sparse_thr=%ld donor_min=%ld free0=%ld "
       "filled=%ld rem_free=%ld donors_try=%ld ok=%ld mean/n=%.4f",
       batch_sz,
       sparse_from_batch,
       static_cast<int64_t>(NMBKM_POST_PCA_SPARSE_ABS_MAX),
       sparse_thr,
       donor_min_cnt,
       n_free0,
       slots_filled,
       static_cast<int64_t>(remaining_free.size()),
       static_cast<int64_t>(donors.size()),
       donors_processed,
       mean_per_center);
   if (labels_owned) {
     ivf_build_mem_ctx_.Deallocate(labels);
   }
   return ret;
 }
 
 // Nested mini-batch k-means (nmbatch), Newling & Fleuret (2016), Algorithm nmbatch.
 // - One Fisher–Yates shuffle defines a fixed order; M_t is the prefix {0,…,b_t−1} in that order.
 // - Each outer iter: zero S_sum / cluster_v / cluster_sse, then Elkan-assign all perm[0..b−1] and re-accumulate
 //   (equivalent to remove+readd on M_{t−1} plus new points; assign_vectors_parallel when |M_t| is large).
 // - Optional starve reinit (compile-time NMBKM_ITER_STARVE_REINIT_ENABLED; off: no iteration-time reinit):
 //   if prefix count for center j is < max(1, floor(b / (k * alpha))), reinit j to w_mu*mu+w_x*x
 //   (mu = mean of largest bucket j_max, x = random full-dataset vector per starving j;
 //   default w_mu=v_max/(v_max+1) (NMBKM_STARVE_RELOCATE_BLEND_MU<0) or fixed w_mu). OB_NMBKM_STARVE_ALPHA when enabled.
 // - sigma/p, rho rule: batch step uses nmbkm_next_batch_size (double until b>=n/2 then +floor(left/2)).
 //   per-point lower bounds l(i,*) are not stored (Elkan recomputed).
 // - HGraph is not used on this path (prefix passes).
 // - Early stop uses prefix mean assign distance vs last iter: (1) if rho keeps b unchanged but change is tiny,
 //   force double b once; if after that larger batch the change is still tiny, stop; (2) once b==n (full prefix),
 //   also stop when change vs previous iter is tiny (same spirit as full-batch k-means), without needing pending.
 int ObElkanKmeansAlgo::do_kmeans_nested_minibatch(const ObIArray<float *> &input_vectors)
 {
   int ret = OB_SUCCESS;
   if (RUNNING_KMEANS != status_) {
     ret = OB_STATE_NOT_MATCH;
     SHARE_LOG(WARN, "status not match", K(ret), K(status_));
   } else {
     const int64_t n = input_vectors.count();
     const int64_t k = kmeans_ctx_->lists_;
     const int64_t dim = kmeans_ctx_->dim_;
     if (n <= 0) {
       ret = OB_INVALID_ARGUMENT;
       SHARE_LOG(WARN, "nested minibatch k-means requires non-empty input", K(ret), K(n));
     } else if (k <= 0 || dim <= 0) {
       ret = OB_INVALID_ARGUMENT;
       SHARE_LOG(WARN, "nested minibatch k-means invalid lists or dim", K(ret), K(k), K(dim));
     } else if (k > INT64_MAX / dim) {
       ret = OB_INVALID_ARGUMENT;
       SHARE_LOG(WARN, "nested minibatch k-means k*dim overflow", K(ret), K(k), K(dim));
     } else if (centers_[cur_idx_].count() != k) {
       ret = OB_ERR_UNEXPECTED;
       SHARE_LOG(WARN, "nested minibatch center buffer size mismatch lists_", K(ret), K(centers_[cur_idx_].count()), K(k));
     }
     int64_t b = 0;
     int64_t nmbkm_div_effective = NMBKM_MIN_N_SCALE;
     if (n > 0) {
       if (NMBKM_FORCE_FULL_BATCH) {
         b = n;
       } else {
         // Initial prefix: n/div; div from env, OB_NMBKM_MIN_N_SCALE_FILE, or /tmp/ob_nmbkm_min_n_scale.
         nmbkm_div_effective = std::max(static_cast<int64_t>(1), nmbkm_min_n_scale_from_env());
         b = std::max(static_cast<int64_t>(1), n / nmbkm_div_effective);
         kmeans_log("kmeans_nmbatch div=%ld b0=%ld n=%ld", nmbkm_div_effective, b, n);
       }
     }
     int64_t b_prev = 0;
     if (OB_SUCC(ret) && NMBKM_FORCE_FULL_BATCH) {
       LOG_INFO("KTS_NMBKM: NMBKM_FORCE_FULL_BATCH on, batch size equals n (full-data assignment each iter)", K(n));
     } else if (OB_SUCC(ret) && n > 0 && !NMBKM_FORCE_FULL_BATCH) {
       LOG_INFO("KTS_NMBKM: initial nested batch b0=n/div (runtime div)", K(b), K(n), K(nmbkm_div_effective));
     }
 
     float *centers_distance = nullptr;
     int32_t *data_cnt_in_cluster = nullptr;
     int64_t *perm = nullptr;
     float *S_sum = nullptr;
     int32_t *cluster_v = nullptr;
     float *cluster_sse = nullptr;
     float *c_old = nullptr;
     int32_t *nearest_scratch = nullptr;
     float *d2_scratch = nullptr;
     int32_t *nmbkm_full_assign_labels = nullptr;
 
     const int64_t center_dis_size = max(1L, k * (k - 1) / 2);
     float *tmp_cd = nullptr;
     if (OB_SUCC(ret)) {
       if (OB_ISNULL(tmp_cd = static_cast<float *>(ivf_build_mem_ctx_.Allocate(sizeof(float) * center_dis_size)))) {
         ret = OB_ALLOCATE_MEMORY_FAILED;
         SHARE_LOG(WARN, "failed to alloc centers_distance", K(ret));
       } else {
         MEMSET(tmp_cd, 0, sizeof(float) * center_dis_size);
         centers_distance = tmp_cd;
       }
     }
     if (OB_SUCC(ret) && OB_ISNULL(data_cnt_in_cluster =
             static_cast<int32_t *>(ivf_build_mem_ctx_.Allocate(sizeof(int32_t) * k)))) {
       ret = OB_ALLOCATE_MEMORY_FAILED;
       SHARE_LOG(WARN, "failed to alloc data_cnt", K(ret));
     } else if (OB_SUCC(ret)) {
       MEMSET(data_cnt_in_cluster, 0, sizeof(int32_t) * k);
     }
     if (OB_SUCC(ret) && OB_ISNULL(perm = static_cast<int64_t *>(ivf_build_mem_ctx_.Allocate(sizeof(int64_t) * n)))) {
       ret = OB_ALLOCATE_MEMORY_FAILED;
       SHARE_LOG(WARN, "failed to alloc perm", K(ret));
     }
     if (OB_SUCC(ret) && OB_ISNULL(S_sum = static_cast<float *>(ivf_build_mem_ctx_.Allocate(sizeof(float) * k * dim)))) {
       ret = OB_ALLOCATE_MEMORY_FAILED;
       SHARE_LOG(WARN, "failed to alloc S_sum", K(ret));
     } else if (OB_SUCC(ret)) {
       MEMSET(S_sum, 0, sizeof(float) * k * dim);
     }
     if (OB_SUCC(ret) && OB_ISNULL(cluster_v = static_cast<int32_t *>(ivf_build_mem_ctx_.Allocate(sizeof(int32_t) * k)))) {
       ret = OB_ALLOCATE_MEMORY_FAILED;
       SHARE_LOG(WARN, "failed to alloc cluster_v", K(ret));
     } else if (OB_SUCC(ret)) {
       MEMSET(cluster_v, 0, sizeof(int32_t) * k);
     }
     if (OB_SUCC(ret) && OB_ISNULL(cluster_sse = static_cast<float *>(ivf_build_mem_ctx_.Allocate(sizeof(float) * k)))) {
       ret = OB_ALLOCATE_MEMORY_FAILED;
       SHARE_LOG(WARN, "failed to alloc cluster_sse", K(ret));
     } else if (OB_SUCC(ret)) {
       MEMSET(cluster_sse, 0, sizeof(float) * k);
     }
     if (OB_SUCC(ret) && OB_ISNULL(c_old = static_cast<float *>(ivf_build_mem_ctx_.Allocate(sizeof(float) * k * dim)))) {
       ret = OB_ALLOCATE_MEMORY_FAILED;
       SHARE_LOG(WARN, "failed to alloc c_old", K(ret));
     }
     if (OB_SUCC(ret) && OB_ISNULL(nearest_scratch = static_cast<int32_t *>(ivf_build_mem_ctx_.Allocate(sizeof(int32_t) * n)))) {
       ret = OB_ALLOCATE_MEMORY_FAILED;
       SHARE_LOG(WARN, "failed to alloc nearest_scratch", K(ret));
     }
     if (OB_SUCC(ret) && OB_ISNULL(d2_scratch = static_cast<float *>(ivf_build_mem_ctx_.Allocate(sizeof(float) * n)))) {
       ret = OB_ALLOCATE_MEMORY_FAILED;
       SHARE_LOG(WARN, "failed to alloc d2_scratch", K(ret));
     }
 
     float starve_alpha_cfg = 0.f;
     if constexpr (NMBKM_ITER_STARVE_REINIT_ENABLED) {
       starve_alpha_cfg = nmbkm_starve_alpha_from_env();
     }
 
     if (OB_SUCC(ret)) {
       for (int64_t i = 0; i < n; ++i) {
         perm[i] = i;
       }
       for (int64_t i = n - 1; i > 0 && OB_SUCC(ret); --i) {
         const int64_t j = ObRandom::rand(0, i);
         if (j >= 0 && j <= i) {
           std::swap(perm[i], perm[j]);
         }
       }
     }
 
     float prev_prefix_mean_assign_distance = 0.f;
     float last_prefix_mean_assign_distance = 0.f;
     bool pending_stall_check_after_forced_expand = false;
     // Forced batch double: looser threshold max(first_stable_diff/10, EARLY); skip first iter after b changes.
     int64_t nmbkm_b_prev_iter_start = -1;
     float nmbkm_first_diff_for_expand = 0.f;
     int64_t nmbkm_b_for_first_diff = -1;
     int64_t nmbkm_total_starved = 0;
     int64_t nmbkm_total_reinit_blend = 0;
     int64_t nmbkm_total_reinit_fb = 0;
 
     if (OB_SUCC(ret) && enable_hgraph_) {
       LOG_INFO("KTS_NMBKM: skip HGraph for nested mini-batch passes; Elkan only", K(b), K(n));
     }
 
     for (int64_t iter = 0; OB_SUCC(ret) && iter < N_ITER; ++iter) {
       if (check_stop()) {
         ret = OB_CANCELED;
         SHARE_LOG(INFO, "kmeans ctx is fore stop", K(ret), K(*this));
         break;
       }
       const int64_t iter_start_time = ObTimeUtility::current_time_ms();
       const int64_t nmbkm_b_start = b;
       int64_t nmbkm_log_starve_thr = INT64_MAX;
       int64_t nmbkm_log_j_max = -1;
       int64_t nmbkm_iter_starved = 0;
       int64_t nmbkm_iter_reinit_blend = 0;
       int64_t nmbkm_iter_reinit_fb = 0;
       float distance = 0.f;
       for (int64_t i = 0; OB_SUCC(ret) && i < k; ++i) {
         for (int64_t j = i + 1; OB_SUCC(ret) && j < k; ++j) {
           if (OB_FAIL(calc_kmeans_distance(centers_[cur_idx_].at(i), centers_[cur_idx_].at(j), dim, distance))) {
             SHARE_LOG(WARN, "failed to calc kmeans distance between centers", K(ret));
           } else {
             set_centers_distance(centers_distance, i, j, distance);
           }
         }
       }
 
       float prefix_dis_sum = 0.f;
       // --- Full prefix M_t: zero stats, then assign all perm[0..b−1] (parallel Elkan when |M_t| large enough) ---
       if (OB_SUCC(ret) && b > 0) {
         MEMSET(S_sum, 0, sizeof(float) * static_cast<size_t>(k) * static_cast<size_t>(dim));
         MEMSET(cluster_v, 0, sizeof(int32_t) * static_cast<size_t>(k));
         MEMSET(cluster_sse, 0, sizeof(float) * static_cast<size_t>(k));
         const bool use_parallel = (OB_NOT_NULL(task_handler_) && max_assign_tasks_ > 0);
         if (use_parallel) {
           centers_[next_idx()].clear();
           MEMSET(data_cnt_in_cluster, 0, sizeof(int32_t) * k);
           ObArray<float *> sub_vecs;
           for (int64_t q = 0; OB_SUCC(ret) && q < b; ++q) {
             if (OB_FAIL(sub_vecs.push_back(input_vectors.at(perm[q])))) {
               SHARE_LOG(WARN, "nmbatch sub_vecs push failed", K(ret));
             }
           }
           float dis_parallel = 0.f;
           if (OB_SUCC(ret) && OB_FAIL(assign_vectors_parallel(sub_vecs, centers_distance, data_cnt_in_cluster, dis_parallel,
                                                              nearest_scratch, d2_scratch))) {
             SHARE_LOG(WARN, "nmbatch parallel assign failed", K(ret));
           } else if (OB_SUCC(ret)) {
             prefix_dis_sum = dis_parallel;
             for (int64_t j = 0; j < k; ++j) {
               MEMCPY(nmbkm_cluster_row(S_sum, dim, j), centers_[next_idx()].at(j), sizeof(float) * static_cast<size_t>(dim));
               cluster_v[j] = data_cnt_in_cluster[j];
             }
             for (int64_t pi = 0; OB_SUCC(ret) && pi < b; ++pi) {
               const int32_t lbl = nearest_scratch[pi];
               if (lbl < 0 || lbl >= k) {
                 ret = OB_ERR_UNEXPECTED;
                 SHARE_LOG(WARN, "nmbatch parallel invalid cluster label", K(ret), K(pi), K(lbl), K(k));
               } else {
                 cluster_sse[lbl] += d2_scratch[pi];
               }
             }
           }
         } else {
           for (int64_t pi = 0; OB_SUCC(ret) && pi < b; ++pi) {
             float *x = input_vectors.at(perm[pi]);
             int64_t new_c = 0;
             float min_dist = 0.f;
             if (OB_FAIL(elkan_find_nearest(x, centers_distance, new_c, min_dist))) {
               SHARE_LOG(WARN, "nmbatch prefix assign failed", K(ret));
             } else if (new_c < 0 || new_c >= k) {
               ret = OB_ERR_UNEXPECTED;
               SHARE_LOG(WARN, "nmbatch prefix invalid cluster id", K(ret), K(new_c), K(k));
             } else {
               const float d2 = min_dist * min_dist;
               const int32_t nc = static_cast<int32_t>(new_c);
               cluster_sse[nc] += d2;
               nmbkm_row_add(nmbkm_cluster_row(S_sum, dim, nc), x, dim);
               cluster_v[nc]++;
               prefix_dis_sum += min_dist;
             }
           }
         }
       }
 
       // c_old, σ̂, p, write new means to next buffer
       float min_sigma_over_p = FLT_MAX;
       if (OB_SUCC(ret)) {
         // If no prefix was processed this iter (b==0), clear assignment stats so we do not reuse stale cluster_v /
         // S_sum from a previous iter (nmbkm_next_batch_size keeps b>=1 for n>0, but guard for robustness).
         if (b == 0) {
           MEMSET(cluster_v, 0, sizeof(int32_t) * static_cast<size_t>(k));
           MEMSET(S_sum, 0, sizeof(float) * static_cast<size_t>(k) * static_cast<size_t>(dim));
           MEMSET(cluster_sse, 0, sizeof(float) * static_cast<size_t>(k));
         }
         for (int64_t j = 0; j < k; ++j) {
           MEMCPY(nmbkm_cluster_row(c_old, dim, j), centers_[cur_idx_].at(j), sizeof(float) * static_cast<size_t>(dim));
         }
         if constexpr (NMBKM_ITER_STARVE_REINIT_ENABLED) {
           const bool starve_reinit_on = (starve_alpha_cfg > 0.f && b > 0);
           const int64_t starve_thr =
               starve_reinit_on ? nmbkm_starve_count_threshold(b, k, starve_alpha_cfg) : INT64_MAX;
           int64_t j_max = 0;
           for (int64_t t = 1; t < k; ++t) {
             if (cluster_v[t] > cluster_v[j_max]) {
               j_max = t;
             }
           }
           const bool have_largest = (cluster_v[j_max] > 0);
           const float *reloc_sum_mx = nullptr;
           float reloc_w_mu = 0.f;
           float reloc_inv_vm = 0.f;
           float reloc_w_x = 0.f;
           if (starve_reinit_on && have_largest) {
             const int32_t vm = cluster_v[j_max];
             reloc_sum_mx = nmbkm_cluster_row(S_sum, dim, j_max);
             reloc_w_mu = nmbkm_starve_relocate_blend_mu_weight(vm);
             reloc_inv_vm = 1.f / static_cast<float>(vm);
             reloc_w_x = 1.f - reloc_w_mu;
           }
           for (int64_t j = 0; OB_SUCC(ret) && j < k; ++j) {
             float *dst = centers_[next_idx()].at(j);
             const int32_t vj = cluster_v[j];
             const bool is_starved = starve_reinit_on && (static_cast<int64_t>(vj) < starve_thr);
             if (is_starved) {
               ++nmbkm_iter_starved;
             }
             const bool relocate = is_starved && have_largest && (vj == 0 || j != j_max);
             if (relocate && n > 0) {
               const int64_t ridx = ObRandom::rand(0, n - 1);
               const float *dup_x = input_vectors.at(ridx);
               for (int64_t d = 0; d < dim; ++d) {
                 const float mu_d = reloc_sum_mx[d] * reloc_inv_vm;
                 dst[d] = reloc_w_mu * mu_d + reloc_w_x * dup_x[d];
               }
               ++nmbkm_iter_reinit_blend;
             } else if (relocate) {
               MEMCPY(dst, centers_[cur_idx_].at(j), sizeof(float) * static_cast<size_t>(dim));
               ++nmbkm_iter_reinit_fb;
             } else {
               nmbkm_fill_center_mean_or_old(
                   dst, nmbkm_cluster_row(S_sum, dim, j), dim, vj, centers_[cur_idx_].at(j));
             }
             if (OB_SUCC(ret) && OB_FAIL(kmeans_ctx_->try_normalize(dim, dst, dst))) {
               LOG_WARN("failed to normalize center", K(ret));
             }
           }
           nmbkm_log_starve_thr = starve_thr;
           nmbkm_log_j_max = have_largest ? j_max : -1;
           nmbkm_total_starved += nmbkm_iter_starved;
           nmbkm_total_reinit_blend += nmbkm_iter_reinit_blend;
           nmbkm_total_reinit_fb += nmbkm_iter_reinit_fb;
         } else {
           for (int64_t j = 0; OB_SUCC(ret) && j < k; ++j) {
             float *dst = centers_[next_idx()].at(j);
             nmbkm_fill_center_mean_or_old(
                 dst, nmbkm_cluster_row(S_sum, dim, j), dim, cluster_v[j], centers_[cur_idx_].at(j));
             if (OB_SUCC(ret) && OB_FAIL(kmeans_ctx_->try_normalize(dim, dst, dst))) {
               LOG_WARN("failed to normalize center", K(ret));
             }
           }
         }
       }
 
       if (OB_SUCC(ret)) {
         constexpr float p_eps = 1e-12f;
         min_sigma_over_p = FLT_MAX;
         for (int64_t j = 0; j < k; ++j) {
           float p_move = 0.f;
           if (OB_FAIL(calc_kmeans_distance(centers_[next_idx()].at(j), nmbkm_cluster_row(c_old, dim, j), dim, p_move))) {
             SHARE_LOG(WARN, "failed calc p(j)", K(ret));
             break;
           }
           if (cluster_v[j] >= 2 && p_move > p_eps) {
             const float vjf = static_cast<float>(cluster_v[j]);
             const float denom = vjf * (vjf - 1.f);
             const float sigma_hat = std::sqrt(std::max(0.f, cluster_sse[j]) / denom);
             const float ratio = sigma_hat / p_move;
             if (ratio < min_sigma_over_p) {
               min_sigma_over_p = ratio;
             }
           }
         }
       }
 
       const float prefix_mean_assign_distance =
           (OB_SUCC(ret) && b > 0) ? (prefix_dis_sum / static_cast<float>(b)) : 0.f;
       constexpr float batch_mean_floor = 1e-20f;
       const float batch_mean_relative_change =
           (iter == 0 || prev_prefix_mean_assign_distance <= batch_mean_floor)
               ? 1.0f
               : (fabsf(prefix_mean_assign_distance - prev_prefix_mean_assign_distance) /
                  prev_prefix_mean_assign_distance);
 
       const bool nmbkm_stable_b = (iter > 0 && nmbkm_b_start == nmbkm_b_prev_iter_start);
       float expand_thr = EARLY_FINISH_THRESHOLD;
       if (nmbkm_stable_b) {
         if (nmbkm_b_for_first_diff != nmbkm_b_start) {
           nmbkm_first_diff_for_expand = batch_mean_relative_change;
           nmbkm_b_for_first_diff = nmbkm_b_start;
         }
         expand_thr = std::max(nmbkm_first_diff_for_expand / 2.f, EARLY_FINISH_THRESHOLD);
       } else {
         nmbkm_b_for_first_diff = -1;
       }
 
       int64_t b_next = b;
       if (OB_SUCC(ret) && !NMBKM_FORCE_FULL_BATCH && b < n) {
         const bool have_ratio = (min_sigma_over_p < FLT_MAX);
         if ((have_ratio && min_sigma_over_p > NMBKM_RHO) || !have_ratio) {
           b_next = nmbkm_next_batch_size(b, n);
         }
       }
       const int64_t b_after_rho_rule = b_next;
       bool forced_expand_for_low_batch_diff = false;
       if (OB_SUCC(ret) && !NMBKM_FORCE_FULL_BATCH && b < n && b_next == b) {
         if (iter > 0 && batch_mean_relative_change <= expand_thr) {
           const int64_t next_b = nmbkm_next_batch_size(b, n);
           if (next_b > b) {
             b_next = next_b;
             forced_expand_for_low_batch_diff = true;
           }
         }
       }
       b_prev = b;
       b = b_next;
       nmbkm_b_prev_iter_start = nmbkm_b_start;
 
       if (OB_SUCC(ret)) {
         bool early_stop_by_loss_threshold = false;
         if (iter > 0 && batch_mean_relative_change <= EARLY_FINISH_THRESHOLD) {
           if (pending_stall_check_after_forced_expand) {
             early_stop_by_loss_threshold = true;
             kmeans_log(
                 "kmeans_nmbatch early_stop batch_diff=%.6f <= thr after forced_expand iter=%ld b_used=%ld",
                 batch_mean_relative_change, iter, b_prev);
           } else if (b_prev == n) {
             early_stop_by_loss_threshold = true;
             kmeans_log(
                 "kmeans_nmbatch early_stop batch_diff=%.6f <= thr full_batch b==n iter=%ld n=%ld",
                 batch_mean_relative_change, iter, n);
           }
         }
         pending_stall_check_after_forced_expand = forced_expand_for_low_batch_diff;
         prev_prefix_mean_assign_distance = prefix_mean_assign_distance;
         last_prefix_mean_assign_distance = prefix_mean_assign_distance;
         // Imbalance of the current prefix M_t only (sum_i cluster_v[i] == b_prev), NOT global n. For global, see cluster_size_stats_global after build.
         const int64_t imbalance_stats_t0_ms = ObTimeUtility::current_time_ms();
         const double prefix_imbalance_factor =
             (b_prev > 0) ? this->calc_imbalance_factor(input_vectors, cluster_v) : 0.0;
         const int64_t imbalance_stats_ms = ObTimeUtility::current_time_ms() - imbalance_stats_t0_ms;
         const float diff = batch_mean_relative_change;
         if (OB_NOT_NULL(kmeans_monitor_)) {
           kmeans_monitor_->set_kmeams_monitor(iter, EARLY_FINISH_THRESHOLD, diff, prefix_imbalance_factor);
         }
         const int64_t iter_cost_ms = ObTimeUtility::current_time_ms() - iter_start_time;
         const float nmbkm_log_first_expand_diff =
             (nmbkm_b_for_first_diff == nmbkm_b_start) ? nmbkm_first_diff_for_expand : -1.f;
         kmeans_log(
             "kmeans_nmbatch iter=%ld iter_cost_ms=%ld imbalance_stats_ms=%ld prefix_mean=%.6f batch_diff=%.6f "
             "expand_thr=%.6f stable_b=%d "
             "first_expand_diff=%.6f b=%ld->%ld rho_would=%ld forced_low_diff=%d early_stop=%d "
             "starve_thr=%ld starve_alpha=%.4f j_max=%ld iter_starved=%ld iter_reinit_blend=%ld iter_reinit_fb=%ld "
             "mu_donor_uses_iter=%ld rand_perturb_iter=%ld cum_starved=%ld cum_reinit_blend=%ld cum_reinit_fb=%ld",
             iter, iter_cost_ms, imbalance_stats_ms, prefix_mean_assign_distance, batch_mean_relative_change, expand_thr,
             nmbkm_stable_b ? 1 : 0, nmbkm_log_first_expand_diff, b_prev, b, b_after_rho_rule,
             forced_expand_for_low_batch_diff ? 1 : 0, early_stop_by_loss_threshold ? 1 : 0,
             nmbkm_log_starve_thr, starve_alpha_cfg, nmbkm_log_j_max, nmbkm_iter_starved, nmbkm_iter_reinit_blend,
             nmbkm_iter_reinit_fb, nmbkm_iter_reinit_blend, nmbkm_iter_reinit_blend, nmbkm_total_starved,
             nmbkm_total_reinit_blend, nmbkm_total_reinit_fb);
 
         cur_idx_ = next_idx();
         if (early_stop_by_loss_threshold) {
           LOG_INFO("finish do kmeans before all iters",
               K(ret),
               K(iter),
               K(prefix_mean_assign_distance),
               K(diff),
               K(prefix_imbalance_factor),
               K(imbalance_stats_ms));
           break;
         } else {
           LOG_INFO("finish one iters",
               K(ret),
               K(iter),
               K(prefix_mean_assign_distance),
               K(diff),
               K(iter_cost_ms),
               K(imbalance_stats_ms));
           if (iter + 1 >= N_ITER) {
             LOG_INFO("finish do kmeans iters",
                 K(ret),
                 K(iter),
                 K(prefix_mean_assign_distance),
                 K(diff),
                 K(prefix_imbalance_factor),
                 K(imbalance_stats_ms));
           }
         }
       }
     }
 
     // Full-data assignment to final centers for per-list count statistics (prefix batches alone do not cover all n).
     // Same nearest-center work as a train iter, but centers are fixed — time is for monitoring only, not center updates.
     // data_cnt_in_cluster is filled in this pass (assign + accumulate_to_centers) and matches nmbkm_full_assign_labels[];
     // nmbkm_postprocess_pca_fill_empty_centers uses the same counts for bucket reserve / sparse vs donor.
     if (OB_SUCC(ret)) {
       centers_[next_idx()].clear();
       MEMSET(data_cnt_in_cluster, 0, sizeof(int32_t) * k);
       if (NMBKM_POST_PCA_FILL_EMPTY_ENABLED
           && OB_ISNULL(nmbkm_full_assign_labels = static_cast<int32_t *>(
                  ivf_build_mem_ctx_.Allocate(sizeof(int32_t) * static_cast<size_t>(n))))) {
         SHARE_LOG(WARN, "NMBKM: alloc full_assign_labels failed; PCA postprocess will reassign if needed", K(n));
       }
       float dis_stats = 0.f;
       const int64_t global_balance_pass_t0_ms = ObTimeUtility::current_time_ms();
       const int ret_stats = search_nearest_center(
           input_vectors, centers_distance, data_cnt_in_cluster, dis_stats, nmbkm_full_assign_labels);
       if (OB_FAIL(ret_stats)) {
         SHARE_LOG(WARN, "full assignment for cluster_size_stats failed", K(ret_stats));
       } else {
         log_cluster_assignment_balance_stats("NMBKM", n, k, data_cnt_in_cluster, "full_n");
         const int64_t global_balance_pass_ms = ObTimeUtility::current_time_ms() - global_balance_pass_t0_ms;
         kmeans_log(
             "cluster_size_stats_global timing global_balance_pass_ms=%ld "
             "(full_n nearest-center assign + imbalance/count stats; no center update) n=%ld k=%ld",
             global_balance_pass_ms,
             n,
             k);
         if (NMBKM_POST_PCA_FILL_EMPTY_ENABLED) {
           const int64_t nmbkm_pca_split_t0_ms = ObTimeUtility::current_time_ms();
           const int ret_pca = nmbkm_postprocess_pca_fill_empty_centers(
               input_vectors, data_cnt_in_cluster, centers_distance, nmbkm_full_assign_labels, b);
           const int64_t nmbkm_pca_split_ms = ObTimeUtility::current_time_ms() - nmbkm_pca_split_t0_ms;
           if (OB_SUCCESS != ret_pca) {
             SHARE_LOG(WARN, "NMBKM postprocess: PCA fill empty centers failed", K(ret_pca));
           }
           kmeans_log(
               "nmbkm_post_pca timing total_ms=%ld ret=%d n=%ld k=%ld "
               "(empty-list split: label/bucket fallback + per-donor PCA+median; 0 if early exit)",
               nmbkm_pca_split_ms,
               ret_pca,
               n,
               k);
           if (OB_SUCC(ret_pca)) {
             centers_[next_idx()].clear();
             MEMSET(data_cnt_in_cluster, 0, sizeof(int32_t) * k);
             float dis_after_pca = 0.f;
             const int64_t after_pca_balance_t0_ms = ObTimeUtility::current_time_ms();
             const int ret_after_pca = search_nearest_center(
                 input_vectors, centers_distance, data_cnt_in_cluster, dis_after_pca, nmbkm_full_assign_labels);
             const int64_t after_pca_balance_ms = ObTimeUtility::current_time_ms() - after_pca_balance_t0_ms;
             if (OB_FAIL(ret_after_pca)) {
               SHARE_LOG(WARN, "NMBKM: full assign after PCA postprocess failed", K(ret_after_pca));
             } else {
               log_cluster_assignment_balance_stats("NMBKM", n, k, data_cnt_in_cluster, "after_pca");
               kmeans_log(
                   "cluster_size_stats_global timing after_pca_balance_pass_ms=%ld "
                   "(full_n nearest-center assign to post-PCA centers + imbalance/count stats) n=%ld k=%ld",
                   after_pca_balance_ms,
                   n,
                   k);
             }
           }
         }
       }
     }
 
 #define NMBKM_FREE_PTR(p)           \
   do {                              \
     if (OB_NOT_NULL(p)) {           \
       ivf_build_mem_ctx_.Deallocate(p); \
       (p) = nullptr;                \
     }                               \
   } while (0)
     NMBKM_FREE_PTR(centers_distance);
     NMBKM_FREE_PTR(data_cnt_in_cluster);
     NMBKM_FREE_PTR(perm);
     NMBKM_FREE_PTR(S_sum);
     NMBKM_FREE_PTR(cluster_v);
     NMBKM_FREE_PTR(cluster_sse);
     NMBKM_FREE_PTR(c_old);
     NMBKM_FREE_PTR(nearest_scratch);
     NMBKM_FREE_PTR(d2_scratch);
     NMBKM_FREE_PTR(nmbkm_full_assign_labels);
 #undef NMBKM_FREE_PTR
     int64_t mem_used = ivf_build_mem_ctx_.get_all_vsag_use_mem_byte() >> 20;
     LOG_INFO("elkan kmeans memused", K(ret), K(mem_used));
     if (OB_SUCC(ret)) {
       kmeans_log(
           "kmeans summary n=%ld lists=%ld dim=%ld final_mean_dis=%.6f mem_mb=%ld "
           "nmbkm_starve_total_starved=%ld nmbkm_starve_total_reinit_blend=%ld nmbkm_starve_total_reinit_fb=%ld "
           "nmbkm_mu_donor_uses_total=%ld nmbkm_rand_perturb_uses_total=%ld "
           "(each blend: one largest-bucket mean j_max + one random sample; fb=memcpy when n==0)",
           input_vectors.count(), kmeans_ctx_->lists_, kmeans_ctx_->dim_, last_prefix_mean_assign_distance, mem_used,
           nmbkm_total_starved, nmbkm_total_reinit_blend, nmbkm_total_reinit_fb, nmbkm_total_reinit_blend,
           nmbkm_total_reinit_blend);
     }
     if (OB_SUCC(ret)) {
       status_ = FINISH;
     }
   }
   return ret;
 }
 
 int ObElkanKmeansAlgo::do_kmeans(const ObIArray<float*> &input_vectors)
 {
   int ret = OB_SUCCESS;
 
   if (RUNNING_KMEANS != status_) {
     ret = OB_STATE_NOT_MATCH;
     SHARE_LOG(WARN, "status not match", K(ret), K(status_));
   } else if (OB_UNLIKELY(kmeans_ctx_->get_train_strategy() == KTS_NMBKM)) {
     ret = do_kmeans_nested_minibatch(input_vectors);
   } else {
     // Upper triangular matrix
     float* centers_distance = nullptr; // half the distance between each two centers
     int32_t *data_cnt_in_cluster = nullptr; // the number of vectors contained in each center (cluster)
     if (!enable_hgraph_) {
       float *tmp = nullptr;
       int64_t center_dis_size = max(1, kmeans_ctx_->lists_ * (kmeans_ctx_->lists_ - 1) / 2);
       if (OB_ISNULL(tmp = static_cast<float *>(ivf_build_mem_ctx_.Allocate(sizeof(float) * center_dis_size)))) {
         ret = OB_ALLOCATE_MEMORY_FAILED;
         SHARE_LOG(WARN, "failed to alloc memory", K(ret), K(ivf_build_mem_ctx_.get_all_vsag_use_mem_byte()));
       } else {
         MEMSET(tmp, 0, sizeof(float) * center_dis_size);
         centers_distance = tmp;
       }
     }
     if (OB_FAIL(ret)) {
     } else if (OB_ISNULL(data_cnt_in_cluster =
         static_cast<int32_t*>(ivf_build_mem_ctx_.Allocate(sizeof(int32_t) * kmeans_ctx_->lists_)))) {
       ret = OB_ALLOCATE_MEMORY_FAILED;
       SHARE_LOG(WARN, "failed to alloc memory", K(ret), K(ivf_build_mem_ctx_.get_all_vsag_use_mem_byte()));
     } else {
       MEMSET(data_cnt_in_cluster, 0, sizeof(int32_t) * kmeans_ctx_->lists_);
     }
 
     const int64_t dim = kmeans_ctx_->dim_;
     float prev_dis_obj = 0;
 
     for (int64_t iter = 0; OB_SUCC(ret) && iter < N_ITER; ++iter) {
 
       if (check_stop()) {
         ret = OB_CANCELED;
         SHARE_LOG(INFO, "kmeans ctx is fore stop", K(ret), K(*this));
         break;
       }
       int64_t iter_start_time = ObTimeUtility::current_time_ms();
       float dis_obj = 0.0;
       MEMSET(data_cnt_in_cluster, 0, sizeof(int32_t) * kmeans_ctx_->lists_);
       centers_[next_idx()].clear();
 
       // try to build hgraph for centers
       if (enable_hgraph_) {
         if (OB_FAIL(build_hgraph_for_centers())) {
           LOG_WARN("failed to build hgraph for centers", K(ret), K(iter));
         }
       }
 
       // 1. search nearest center
       if (OB_FAIL(search_nearest_center(input_vectors, centers_distance, data_cnt_in_cluster, dis_obj))) {
         SHARE_LOG(WARN, "failed to search nearest center", K(ret));
       }
 
       // 2. calc the new centers
       for (int64_t i = 0; OB_SUCC(ret) && i < kmeans_ctx_->lists_; ++i) {
         if (data_cnt_in_cluster[i] > 0) {
           if (OB_FAIL(centers_[next_idx()].divide(i, data_cnt_in_cluster[i]))) {
             SHARE_LOG(WARN, "failed to divide vector", K(ret));
           }
         } else {
           // use random sample vector as the center
           int64_t random = 0;
           const int64_t sample_cnt = input_vectors.count();
           random = ObRandom::rand(0, sample_cnt - 1);
           if (OB_FAIL(centers_[next_idx()].add(i, kmeans_ctx_->dim_, input_vectors.at(random)))) {
             SHARE_LOG(WARN, "failed to add vector", K(ret));
           }
         }
         // 3. normalize the new center, if need
         if (OB_SUCC(ret)) {
           if (OB_FAIL(kmeans_ctx_->try_normalize(
               kmeans_ctx_->dim_,
               centers_[next_idx()].at(i),
               centers_[next_idx()].at(i)))) {
             LOG_WARN("failed to normalize vector", K(ret));
           }
         }
       } // end for
 
       // // iter==0：对「当前 cur_idx_（本轮 assign 用的旧中心）」再扫一遍全量 Elkan 距离和；与 dis_obj 应对同一组中心。
       // // enable_hgraph_ 时未分配 centers_distance，不能走 Elkan 全量扫，此处跳过（dis_obj 仍为 HGraph assign 的均值）。
       // if (iter == 0 && OB_NOT_NULL(centers_distance)) {
       //   const int64_t sample_n = input_vectors.count();
       //   if (sample_n > 0) {
       //     float full_dis_sum = 0.f;
       //     if (OB_SUCC(ret) && OB_FAIL(assign_vectors_parallel(input_vectors, centers_distance, nullptr, full_dis_sum,
       //                                                         nullptr, nullptr, false, false))) {
       //       SHARE_LOG(WARN, "elkan iter0 full_dataset loss: assign sum failed", K(ret));
       //     } else if (OB_SUCC(ret)) {
       //       const float mean_dis = full_dis_sum / static_cast<float>(sample_n);
       //       kmeans_log(
       //           "kmeans_elkan iter0 full_dataset_mean_assign_dis=%.6f n=%ld (cf dis_obj=%.6f)", mean_dis, sample_n,
       //           dis_obj);
       //     }
       //   }
       // }
 
       // 4. check finish && switch center buffer
       if (OB_SUCC(ret)) {
         const int64_t imbalance_stats_t0_ms = ObTimeUtility::current_time_ms();
         double imbalance_factor = this->calc_imbalance_factor(input_vectors, data_cnt_in_cluster);
         const int64_t imbalance_stats_ms = ObTimeUtility::current_time_ms() - imbalance_stats_t0_ms;
         float diff = (iter == 0) ? 1.0 : fabs(prev_dis_obj - dis_obj) / prev_dis_obj;
         prev_dis_obj = dis_obj;
         if (OB_NOT_NULL(kmeans_monitor_)) {
           kmeans_monitor_->set_kmeams_monitor(iter, EARLY_FINISH_THRESHOLD, diff, imbalance_factor);
         }
         if (iter > 0 && diff <= EARLY_FINISH_THRESHOLD) {
           const int64_t iter_cost_ms = ObTimeUtility::current_time_ms() - iter_start_time;
           LOG_INFO(
               "finish do kmeans before all iters", K(ret), K(iter), K(dis_obj), K(diff), K(imbalance_factor), K(imbalance_stats_ms));
           kmeans_log(
               "kmeans iter iter=%ld iter_cost_ms=%ld imbalance_stats_ms=%ld loss=%.6f diff=%.6f early_stop=1",
               iter,
               iter_cost_ms,
               imbalance_stats_ms,
               dis_obj,
               diff);
           // Same as non-early path: new means live in centers_[next_idx()] until we swap.
           cur_idx_ = next_idx();
           break;  // finish
         } else {
           cur_idx_ = next_idx();
           const int64_t iter_cost_ms = ObTimeUtility::current_time_ms() - iter_start_time;
           LOG_INFO("finish one iters", K(ret), K(iter), K(dis_obj), K(diff), K(iter_cost_ms), K(imbalance_stats_ms));
           kmeans_log(
               "kmeans iter iter=%ld iter_cost_ms=%ld imbalance_stats_ms=%ld loss=%.6f diff=%.6f",
               iter,
               iter_cost_ms,
               imbalance_stats_ms,
               dis_obj,
               diff);
           if (iter + 1 >= N_ITER) {
             LOG_INFO("finish do kmeans iters", K(ret), K(iter), K(dis_obj), K(diff), K(imbalance_factor), K(imbalance_stats_ms));
           }
         }
       }
     }  // iter end for
 
     // only need to release index when hgraph is enabled
     if (enable_hgraph_ && OB_NOT_NULL(hgraph_index_)) {
       obvectorutil::delete_index(hgraph_index_);
       hgraph_index_ = nullptr;
       LOG_DEBUG("Released HGraph index after K-means completion");
     }
 
     // free tmp memory
     int64_t mem_used = ivf_build_mem_ctx_.get_all_vsag_use_mem_byte() >> 20;
     LOG_INFO("elkan kmeans memused", K(ret), K(mem_used));
     if (OB_SUCC(ret)) {
       kmeans_log(
           "kmeans summary n=%ld lists=%ld dim=%ld final_mean_dis=%.6f mem_mb=%ld",
           input_vectors.count(), kmeans_ctx_->lists_, kmeans_ctx_->dim_, prev_dis_obj, mem_used);
     }
     if (OB_SUCC(ret) && OB_NOT_NULL(data_cnt_in_cluster)) {
       log_cluster_assignment_balance_stats("KM", input_vectors.count(), kmeans_ctx_->lists_, data_cnt_in_cluster, "full_n");
     }
     if (OB_NOT_NULL(centers_distance)) {
       ivf_build_mem_ctx_.Deallocate(centers_distance);
       centers_distance = nullptr;
     }
     if (OB_NOT_NULL(data_cnt_in_cluster)) {
       ivf_build_mem_ctx_.Deallocate(data_cnt_in_cluster);
       data_cnt_in_cluster = nullptr;
     }
     if (OB_SUCC(ret)) {
       status_ = FINISH;
     }
   }
   return ret;
 }
 
 int ObElkanKmeansAlgo::add_vector_to_center_safe(int64_t center_idx, int64_t dim, float* vector, int32_t* data_cnt_in_cluster)
 {
   int ret = OB_SUCCESS;
   common::ObSpinLockGuard guard(assign_lock_);
   if (OB_FAIL(get_centers(next_idx()).add(center_idx, dim, vector))) {
     SHARE_LOG(WARN, "failed to add vector to center buffer", K(ret));
   } else {
     ++data_cnt_in_cluster[center_idx];
   }
   return ret;
 }
 
 int ObElkanKmeansAlgo::find_nearest_center_with_hgraph(const float* vector, int64_t &nearest_center_idx, float &distance)
 {
   int ret = OB_SUCCESS;
   nearest_center_idx = -1;
   distance = std::numeric_limits<float>::max();
   ObVsagSearchAlloc vsag_allocator(kmeans_ctx_->tenant_id_);
 
   if (OB_ISNULL(vector)) {
     ret = OB_INVALID_ARGUMENT;
     LOG_WARN("vector is null", K(ret));
   } else if (OB_ISNULL(hgraph_index_)) {
     ret = OB_ERR_NULL_VALUE;
     LOG_WARN("hgraph index not built, fallback to brute force", K(ret));
   } else {
     const float* distances = nullptr;
     const int64_t* result_ids = nullptr;
     int64_t result_size = 0;
     const char* extra_info = nullptr;
     int ef_search = 64;
 
     if (OB_FAIL(obvectorutil::knn_search(
         hgraph_index_,
         const_cast<float*>(vector),
         kmeans_ctx_->dim_,
         1,
         distances,
         result_ids,
         extra_info,
         result_size,
         ef_search,
         nullptr, // invalid filter
         false,   // reverse_filter
         false,   // use_extra_info_filter
         1.0f,    // valid_ratio
         &vsag_allocator,
         false))) { // need_extra_info
       LOG_WARN("HGraph knn search failed", K(ret));
     } else if (result_size <= 0 || OB_ISNULL(result_ids) || OB_ISNULL(distances)) {
       ret = OB_ERR_UNEXPECTED;
       LOG_WARN("Invalid HGraph search result", K(ret), K(result_size), KP(result_ids), KP(distances));
     } else if (result_ids[0] < 1 || result_ids[0] > kmeans_ctx_->lists_) {
       ret = OB_ERR_UNEXPECTED;
       LOG_WARN("HGraph result out of range", K(ret), K(result_ids[0]), K(kmeans_ctx_->lists_));
     } else {
       nearest_center_idx = result_ids[0] - 1;
       distance = distances[0];
       LOG_DEBUG("found nearest center with hgraph", K(nearest_center_idx), K(distance));
     }
   }
 
   return ret;
 }
 
 int ObElkanKmeansAlgo::elkan_find_nearest(const float *sample_vector, float *centers_distance,
                                           int64_t &nearest_center_idx, float &min_distance)
 {
   int ret = OB_SUCCESS;
   const int64_t dim = kmeans_ctx_->dim_;
   const int64_t center_count = kmeans_ctx_->lists_;
   if (OB_ISNULL(centers_distance)) {
     ret = OB_INVALID_ARGUMENT;
     SHARE_LOG(WARN, "centers_distance is null", K(ret));
   } else if (center_count <= 0 || dim <= 0) {
     ret = OB_INVALID_ARGUMENT;
     SHARE_LOG(WARN, "elkan_find_nearest invalid center_count or dim", K(ret), K(center_count), K(dim));
   } else if (OB_FAIL(calc_kmeans_distance(sample_vector, centers_[cur_idx_].at(0), dim, min_distance))) {
     SHARE_LOG(WARN, "failed to calc kmeans distance", K(ret));
   } else {
     nearest_center_idx = 0;
     float gate_distance = min_distance * GATE_DISTANCE_FACTOR;
     for (int64_t j = 1; OB_SUCC(ret) && j < center_count; ++j) {
       const float dis_near_cur = get_centers_distance(centers_distance, nearest_center_idx, j);
       if (dis_near_cur < gate_distance) {
         float dis_half_dim = 0.0f;
         if (OB_FAIL(calc_kmeans_distance(sample_vector, centers_[cur_idx_].at(j), dim / 2, dis_half_dim))) {
           SHARE_LOG(WARN, "failed to calc kmeans distance", K(ret));
         } else if (dis_half_dim < min_distance) {
           float full_distance = 0.0f;
           if (OB_FAIL(calc_kmeans_distance(sample_vector + dim / 2, centers_[cur_idx_].at(j) + dim / 2,
                                            dim - dim / 2, full_distance))) {
             SHARE_LOG(WARN, "failed to calc kmeans distance", K(ret));
           } else {
             full_distance += dis_half_dim;
             if (full_distance < min_distance) {
               min_distance = full_distance;
               gate_distance = min_distance * GATE_DISTANCE_FACTOR;
               nearest_center_idx = j;
             }
           }
         }
       }
     }
   }
   return ret;
 }
 
 int ObElkanKmeansAlgo::assign_vectors_range(const ObIArray<float *> &input_vectors, int64_t start_idx, int64_t end_idx,
                                             float *centers_distance, int32_t *data_cnt_in_cluster, float &dis_obj,
                                             bool use_safe_add,
                                             int32_t *nearest_label_out,
                                             float *min_d2_out,
                                             const bool accumulate_to_centers,
                                             const bool allow_hgraph_assign)
 {
   int ret = OB_SUCCESS;
   const int64_t dim = kmeans_ctx_->dim_;
   const int64_t center_count = kmeans_ctx_->lists_;
   const bool use_hgraph = allow_hgraph_assign && is_hgraph_available();
   if (!use_hgraph && OB_ISNULL(centers_distance)) {
     ret = OB_INVALID_ARGUMENT;
     SHARE_LOG(WARN, "centers distance is required without hgraph", K(ret));
   }
 
   for (int64_t i = start_idx; OB_SUCC(ret) && i < end_idx; ++i) {
     if (check_stop()) {
       ret = OB_CANCELED;
       SHARE_LOG(WARN, "check stop", K(ret));
       break;
     }
     float *sample_vector = input_vectors.at(i);
     int64_t nearest_center_idx = 0;
     float min_distance = FLT_MAX;
 
     if (use_hgraph) {
       if (OB_FAIL(find_nearest_center_with_hgraph(sample_vector, nearest_center_idx, min_distance))) {
         SHARE_LOG(WARN, "failed to find nearest center with hgraph", K(ret));
       } else if (nearest_center_idx < 0 || nearest_center_idx >= center_count) {
         ret = OB_ERR_UNEXPECTED;
         SHARE_LOG(WARN, "invalid hgraph result", K(ret), K(nearest_center_idx), K(center_count));
       }
     } else if (OB_FAIL(elkan_find_nearest(sample_vector, centers_distance, nearest_center_idx, min_distance))) {
       SHARE_LOG(WARN, "failed to find nearest center (Elkan)", K(ret));
     }
     if (OB_SUCC(ret)) {
       dis_obj += min_distance;
       if (nullptr != nearest_label_out) {
         nearest_label_out[i] = static_cast<int32_t>(nearest_center_idx);
       }
       if (nullptr != min_d2_out) {
         min_d2_out[i] = min_distance * min_distance;
       }
       if (accumulate_to_centers) {
         if (use_safe_add) {
           if (OB_FAIL(add_vector_to_center_safe(nearest_center_idx, dim, sample_vector, data_cnt_in_cluster))) {
             SHARE_LOG(WARN, "failed to add vector to center buffer safely", K(ret));
           }
         } else {
           if (OB_FAIL(centers_[next_idx()].add(nearest_center_idx, dim, sample_vector))) {
             SHARE_LOG(WARN, "failed to add vector to center buffer", K(ret));
           } else {
             ++data_cnt_in_cluster[nearest_center_idx];
           }
         }
       }
     }
   }
 
   return ret;
 }
 
 int ObKmeansAlgo::calc_distances_range(const ObIArray<float*> &input_vectors, int64_t start_idx, int64_t end_idx,
                                       float* current_center, float* weight, const int64_t dim, float &sum)
 {
   int ret = OB_SUCCESS;
 
   for (int64_t i = start_idx; OB_SUCC(ret) && i < end_idx; ++i) {
     float distance = 0.0f;
     if (OB_FAIL(ObKmeansAlgo::calc_kmeans_distance(input_vectors.at(i), current_center, dim, distance))) {
       SHARE_LOG(WARN, "failed to calc kmeans distance", K(ret));
     } else {
       distance *= distance;
       if (distance < weight[i]) {
         weight[i] = distance;
       }
       sum += weight[i];
     }
   }
 
   return ret;
 }
 
 // ------------------ ObIvfBuildHelper implement ------------------
 void ObIvfBuildHelper::reset()
 {
   ObIAllocator *allocator = get_allocator();
   if (OB_NOT_NULL(allocator) && OB_NOT_NULL(ivf_build_mem_ctx_)) {
     ivf_build_mem_ctx_->~ObIvfMemContext();
     allocator->free(ivf_build_mem_ctx_);
     ivf_build_mem_ctx_ = nullptr;
   }
 }
 
 int ObIvfBuildHelper::init(ObString &init_str, lib::MemoryContext &parent_mem_ctx, uint64_t *all_vsag_use_mem)
 {
   int ret = OB_SUCCESS;
   if (OB_FAIL(ObVectorIndexUtil::parser_params_from_string(init_str, ObVectorIndexType::VIT_IVF_INDEX, param_))) {
     LOG_WARN("failed to parse params.", K(ret));
   } else if (OB_ISNULL(ivf_build_mem_ctx_ = OB_NEWx(ObIvfMemContext, get_allocator(), all_vsag_use_mem))) {
     ret = OB_ALLOCATE_MEMORY_FAILED;
     LOG_WARN("failed to create ivf_build_mem_ctx", K(ret));
   } else if (OB_FAIL(ivf_build_mem_ctx_->init(parent_mem_ctx, all_vsag_use_mem, tenant_id_, ObIvfMemContext::IVF_BUILD_LABEL))) {
     LOG_WARN("failed to init memory context", K(ret));
     get_allocator()->free(ivf_build_mem_ctx_);
     ivf_build_mem_ctx_ = nullptr;
   } else {
     int64_t mem_used = ivf_build_mem_ctx_->get_all_vsag_use_mem_byte() >> 20;
     SHARE_LOG(INFO, "init ivf_build_mem_ctx", K(ret), K(mem_used));
   }
   return ret;
 }
 
 int ObIvfBuildHelper::init_ctx(int64_t dim)
 {
   int ret = OB_SUCCESS;
   lib::ObMutexGuard guard(lock_);
   if (is_inited_) {
     ret = OB_SUCCESS;
     SHARE_LOG(INFO, "init ctx already inited", K(ret), K(dim), K(param_));
   } else if (first_ret_code_ != OB_SUCCESS) {
     ret = first_ret_code_;
     SHARE_LOG(WARN, "init falied before", K(ret), K(dim), K(param_));
   } else if (OB_FAIL(init_kmeans_ctx(dim))) {
     SHARE_LOG(WARN, "failed to init kmeans ctx", K(ret), K(dim), K(param_));
   } else {
     is_inited_ = true;
   }
   first_ret_code_ = ret;
 
   return ret;
 }
 
 void ObIvfBuildHelper::inc_ref()
 {
   ATOMIC_INC(&ref_cnt_);
   OB_LOG(DEBUG, "inc ref count", K(ref_cnt_), KP(this), KPC(this), K(lbt())); // remove later
 }
 
 bool ObIvfBuildHelper::dec_ref_and_check_release()
 {
   int64_t ref_count = ATOMIC_SAF(&ref_cnt_, 1);
   OB_LOG(DEBUG,"dec ref count", K(ref_count), KP(this), KPC(this), K(lbt())); // remove later
   return (ref_count == 0);
 }
 
 int64_t ObIvfBuildHelper::get_free_vector_mem_size()
 {
   int ret = OB_SUCCESS;
   int64_t free_vector_mem_size = 0;
   int64_t tenant_mem_size = 0;
   int64_t curr_used = 0;
   if (OB_ISNULL(ivf_build_mem_ctx_)) {
     ret = OB_ERR_UNEXPECTED;
     LOG_WARN("mem ctx is null", K(ret));
   } else if (OB_FALSE_IT(curr_used = ATOMIC_LOAD(ivf_build_mem_ctx_->get_all_vsag_use_mem()))) {
   } else if (OB_FAIL(ObPluginVectorIndexHelper::get_vector_memory_limit_size(tenant_id_, tenant_mem_size))) {
     LOG_WARN("failed to get vector mem limit size.", K(ret), K(tenant_id_));
   } else if (tenant_mem_size > curr_used) {
     free_vector_mem_size = tenant_mem_size - curr_used;
   }
   LOG_INFO("free vector mem limit size.", K(ret), K(free_vector_mem_size), K(tenant_mem_size), K(curr_used));
   return free_vector_mem_size;
 }
 
 // ------------------ ObIvfFlatBuildHelper implement ------------------
 ObIvfFlatBuildHelper::~ObIvfFlatBuildHelper()
 {
   if (OB_NOT_NULL(executor_) && OB_NOT_NULL(ivf_build_mem_ctx_)) {
     executor_->~ObSingleKmeansExecutor();
     ivf_build_mem_ctx_->Deallocate(executor_);
     executor_ = nullptr;
   }
 }
 
 int ObIvfFlatBuildHelper::init_kmeans_ctx(const int64_t dim)
 {
   int ret = OB_SUCCESS;
   ObKmeansAlgoType algo_type = ob_resolve_kmeans_algo_type_from_env(false /* ivf coarse */);
   void *buf = nullptr;
   ObVectorNormalizeInfo *norm_info = nullptr;
   if (OB_NOT_NULL(executor_)) {
     // do nothing
   } else if (0 >= param_.nlist_ || 0 >= param_.sample_per_nlist_ || 0 >= dim || VIDA_MAX <= param_.dist_algorithm_) {
     ret = OB_INVALID_ARGUMENT;
     SHARE_LOG(WARN, "invalid argument", K(ret), K(dim), K(param_));
   } else if ((VIDA_IP == param_.dist_algorithm_ || VIDA_COS == param_.dist_algorithm_) &&
               FALSE_IT(norm_info = &norm_info_)) { // IP and COS algorithms need normalization
   } else if (OB_ISNULL(ivf_build_mem_ctx_)) {
     ret = OB_NOT_INIT;
     SHARE_LOG(WARN, "ivf_build_mem_ctx_ is null", K(ret));
   } else {
     void *tmp_buf = nullptr;
     if (OB_ISNULL(tmp_buf = ivf_build_mem_ctx_->Allocate(sizeof(ObSingleKmeansExecutor)))) {
       ret = OB_ALLOCATE_MEMORY_FAILED;
       LOG_WARN("failed to alloc tmp_buf", K(ret), K(ivf_build_mem_ctx_->get_all_vsag_use_mem_byte()));
     } else if (OB_FALSE_IT(executor_ = new (tmp_buf) ObSingleKmeansExecutor(*ivf_build_mem_ctx_))) {
     } else if (OB_FAIL(executor_->init(algo_type, tenant_id_, param_.nlist_,
                                        param_.sample_per_nlist_, dim, param_.dist_algorithm_, norm_info))) {
       LOG_WARN("failed to init kmeans ctx", K(ret));
     } else {
       is_inited_ = true;
     }
   }
   first_ret_code_ = ret;
   return ret;
 }
 
 // ------------------ ObIvfSq8BuildHelper implement ------------------
 ObIvfSq8BuildHelper::~ObIvfSq8BuildHelper()
 {
   if (OB_NOT_NULL(ivf_build_mem_ctx_)) {
     if (OB_NOT_NULL(min_vector_)) {
       ivf_build_mem_ctx_->Deallocate(min_vector_);
       min_vector_ = nullptr;
     }
     if (OB_NOT_NULL(max_vector_)) {
       ivf_build_mem_ctx_->Deallocate(max_vector_);
       max_vector_ = nullptr;
     }
     if (OB_NOT_NULL(step_vector_)) {
       ivf_build_mem_ctx_->Deallocate(step_vector_);
       step_vector_ = nullptr;
     }
   }
 }
 
 int ObIvfSq8BuildHelper::update(const float *vector, int64_t dim)
 {
   int ret = OB_SUCCESS;
   if (!is_inited_) {
     ret = OB_NOT_INIT;
     LOG_WARN("ObIvfSq8BuildHelper is not inited", K(ret));
   } else if (OB_ISNULL(vector) || dim != dim_) {
     ret = OB_INVALID_ARGUMENT;
     LOG_WARN("invalid vector or dim", K(ret), KP(vector), K(dim), K(dim_));
   }
   float cur = 0;
   for (int i = 0; i < dim_ && OB_SUCC(ret); ++i) {
     cur = vector[i];
     if (cur < min_vector_[i]) {
       min_vector_[i] = cur;
     }
     if (cur > max_vector_[i]) {
       max_vector_[i] = cur;
     }
   }
   return ret;
 }
 
 int ObIvfSq8BuildHelper::build()
 {
   int ret = OB_SUCCESS;
   if (!is_inited_) {
     ret = OB_NOT_INIT;
     LOG_WARN("ObIvfSq8BuildHelper is not inited", K(ret));
   } else {
     for (int i = 0; i < dim_; ++i) {
       step_vector_[i] = (max_vector_[i] - min_vector_[i]) / ObIvfConstant::SQ8_META_STEP_SIZE;
     }
   }
   return ret;
 }
 
 int ObIvfSq8BuildHelper::get_result(int row_pos, float *&vector)
 {
   int ret = OB_SUCCESS;
   if (!is_inited_) {
     ret = OB_NOT_INIT;
     LOG_WARN("ObIvfSq8BuildHelper is not inited", K(ret));
   } else if (row_pos < 0 || row_pos >= ObIvfConstant::SQ8_META_ROW_COUNT) {
     ret = OB_INVALID_ARGUMENT;
     LOG_WARN("row_pos out of range", K(ret), K(row_pos));
   } else if (row_pos == ObIvfConstant::SQ8_META_MIN_IDX) {
     vector = min_vector_;
   } else if (row_pos == ObIvfConstant::SQ8_META_MAX_IDX) {
     vector = max_vector_;
   } else if (row_pos == ObIvfConstant::SQ8_META_STEP_IDX) {
     vector = step_vector_;
   } else {
     // should not be here
     ret = OB_ERR_UNEXPECTED;
     LOG_WARN("unexpected row pos", K(ret), K(row_pos));
   }
   return ret;
 }
 
 int ObIvfSq8BuildHelper::init_kmeans_ctx(const int64_t vec_dim)
 {
   int ret = OB_SUCCESS;
   if (OB_ISNULL(ivf_build_mem_ctx_)) {
     ret = OB_NOT_INIT;
     SHARE_LOG(WARN, "ivf_build_mem_ctx_ is null", K(ret));
   } else if (OB_ISNULL(min_vector_ = static_cast<float*>(ivf_build_mem_ctx_->Allocate(vec_dim * sizeof(float))))) {
     ret = OB_ALLOCATE_MEMORY_FAILED;
     SHARE_LOG(WARN, "failed to alloc memory", K(ret), K(vec_dim), K(ivf_build_mem_ctx_->get_all_vsag_use_mem_byte()));
   } else if (OB_ISNULL(max_vector_ = static_cast<float*>(ivf_build_mem_ctx_->Allocate(vec_dim * sizeof(float))))) {
     ret = OB_ALLOCATE_MEMORY_FAILED;
     SHARE_LOG(WARN, "failed to alloc memory", K(ret), K(vec_dim), K(ivf_build_mem_ctx_->get_all_vsag_use_mem_byte()));
   } else if (OB_ISNULL(step_vector_ = static_cast<float*>(ivf_build_mem_ctx_->Allocate(vec_dim * sizeof(float))))) {
     ret = OB_ALLOCATE_MEMORY_FAILED;
     SHARE_LOG(WARN, "failed to alloc memory", K(ret), K(vec_dim), K(ivf_build_mem_ctx_->get_all_vsag_use_mem_byte()));
   } else {
     MEMSET(step_vector_, 0, sizeof(float) * vec_dim);
     for (int i = 0; i < vec_dim; ++i) {
       min_vector_[i] = FLT_MAX;
       max_vector_[i] = FLT_MIN;
     }
     dim_ = vec_dim;
     is_inited_ = true;
   }
   first_ret_code_ = ret;
   return ret;
 }
 
 // ------------------ ObIvfPqBuildHelper implement ------------------
 ObIvfPqBuildHelper::~ObIvfPqBuildHelper()
 {
   if (OB_NOT_NULL(executor_) && OB_NOT_NULL(ivf_build_mem_ctx_)) {
     executor_->~ObMultiKmeansExecutor();
     ivf_build_mem_ctx_->Deallocate(executor_);
     executor_ = nullptr;
   }
 }
 
 int ObIvfPqBuildHelper::init_kmeans_ctx(const int64_t dim)
 {
   int ret = OB_SUCCESS;
   ObKmeansAlgoType algo_type = ob_resolve_kmeans_algo_type_from_env(true /* pq codebook */);

   void *buf = nullptr;
   int64_t pqnlist = 0;
   int64_t sample_per_nlist = 0;
   if (OB_NOT_NULL(executor_)) {
     // do nothing
   } else if (0 >= param_.nbits_ || 24 < param_.nbits_ || 0 >= param_.sample_per_nlist_ || 0 >= dim || VIDA_MAX <= param_.dist_algorithm_
             || param_.m_ <= 0) {
     ret = OB_INVALID_ARGUMENT;
     SHARE_LOG(WARN, "invalid argument", K(ret), K(dim), K(param_));
   } else if (FALSE_IT(pqnlist = 1L << param_.nbits_)) {
   } else if (OB_ISNULL(ivf_build_mem_ctx_)) {
     ret = OB_NOT_INIT;
     SHARE_LOG(WARN, "ivf_build_mem_ctx_ is null", K(ret));
   } else {
     int64_t sample_count = MAX(pqnlist * param_.sample_per_nlist_, param_.nlist_ * param_.sample_per_nlist_);
     sample_per_nlist = sample_count / pqnlist;
     void *tmp_buf = nullptr;
     if (OB_ISNULL(tmp_buf = ivf_build_mem_ctx_->Allocate(sizeof(ObMultiKmeansExecutor)))) {
       ret = OB_ALLOCATE_MEMORY_FAILED;
       LOG_WARN("failed to alloc tmp_buf", K(ret), K(ivf_build_mem_ctx_->get_all_vsag_use_mem_byte()));
     } else if (OB_FALSE_IT(executor_ = new (tmp_buf) ObMultiKmeansExecutor(*ivf_build_mem_ctx_))) {
     } else if (OB_FAIL(executor_->init(algo_type,
                                   tenant_id_,
                                   pqnlist,
                                   sample_per_nlist,
                                   dim,
                                   param_.dist_algorithm_,
                                   nullptr, // pq center kmeans no need normlize, Reference faiss
                                   param_.m_))) {
       LOG_WARN("failed to init kmeans ctx", K(ret), K(param_), K(pqnlist));
     } else {
       is_inited_ = true;
     }
   }
   first_ret_code_ = ret;
   return ret;
 }
 
 int ObIvfPqBuildHelper::build(const common::ObTableID &table_id, const common::ObTabletID &tablet_id, ObInsertMonitor* insert_monitor)
 {
   int ret = OB_SUCCESS;
   if (OB_ISNULL(executor_)) {
     ret = OB_ERR_UNEXPECTED;
     LOG_WARN("unexpected nullptr ctx", K(ret));
   } else if (can_use_parallel()) {
     if (OB_FAIL(executor_->build_parallel(table_id, tablet_id, insert_monitor))) {
       LOG_WARN("failed to build clusters", K(ret));
     }
   } else if (OB_FAIL(executor_->build(insert_monitor))) {
     LOG_WARN("failed to build clusters", K(ret));
   }
 
 #ifdef ERRSIM
   if (OB_SUCC(ret)) {
     ret = OB_E(common::EventTable::EN_VEC_INDEX_IVF_PQ_BUILD_ERR) OB_SUCCESS;
     if (OB_FAIL(ret)) {
       LOG_WARN("[ERRSIM] fail to build ivf pq", K(ret));
     }
   }
 #endif
 
   return ret;
 }
 
 bool ObIvfPqBuildHelper::can_use_parallel()
 {
   int ret = OB_SUCCESS;
   bool res = false;
   int64_t max_thread_cnt = MTL_CPU_COUNT() * ObKmeansBuildTaskHandler::THREAD_FACTOR;
   max_thread_cnt = OB_MAX(max_thread_cnt, ObKmeansBuildTaskHandler::MIN_THREAD_COUNT);
   uint64_t parallel_need_max_mem = 0;
   int64_t vector_free_mem = 0;
   if (OB_ISNULL(executor_)) {
     ret = OB_ERR_UNEXPECTED;
     LOG_WARN("executor_ is null", KR(ret));
   } else if (OB_FAIL(ObVectorIndexUtil::estimate_ivf_pq_kmeans_memory(executor_->get_max_sample_count(), param_, max_thread_cnt, parallel_need_max_mem))) {
     LOG_WARN("estimate ivf memory failed", KR(ret), K(executor_->get_max_sample_count()), K(param_));
   } else {
     vector_free_mem = get_free_vector_mem_size();
     if (vector_free_mem > parallel_need_max_mem) {
       res = true;
     }
   }
   LOG_INFO("can use parallel", K(res), K(max_thread_cnt), K(parallel_need_max_mem), K(vector_free_mem));
   return res;
 }
 
 /**************************** ObKmeansBuildTaskHandler ******************************/
 int ObKmeansBuildTaskHandler::init(int tg_id)
 {
   int ret = OB_SUCCESS;
   if (is_inited_) {
     LOG_INFO("init before", KR(ret));
   } else if (INVALID_TG_ID == tg_id) {
     ret = OB_INVALID_ARGUMENT;
     LOG_WARN("invalid tg_id", KR(ret), K(tg_id));
   } else {
     tg_id_ = tg_id;
     is_inited_ = true;
     LOG_INFO("init vector kmeans build task handler", K(ret), K_(tg_id));
   }
   return ret;
 }
 
 int ObKmeansBuildTaskHandler::start()
 {
   int ret = OB_SUCCESS;
   int64_t max_thread_cnt = MTL_CPU_COUNT() * THREAD_FACTOR;
   max_thread_cnt = OB_MAX(max_thread_cnt, MIN_THREAD_COUNT);
   if (IS_NOT_INIT) {
     ret = OB_NOT_INIT;
     LOG_WARN("handler is not init", KR(ret));
   } else if (OB_FAIL(TG_SET_ADAPTIVE_THREAD(tg_id_, MIN_THREAD_COUNT,
                                             max_thread_cnt))) {  // must be call TG_SET_ADAPTIVE_THREAD
     LOG_WARN("TG_SET_ADAPTIVE_THREAD failed", KR(ret), K_(tg_id));
   } else if (OB_FAIL(TG_SET_HANDLER_AND_START(tg_id_, *this))) {
     LOG_WARN("TG_SET_HANDLER_AND_START failed", KR(ret), K_(tg_id));
   } else {
     max_thread_cnt_ = max_thread_cnt;
     LOG_INFO("succ to start vector kmeans build task handler", K_(tg_id), K(max_thread_cnt));
   }
   return ret;
 }
 
 void ObKmeansBuildTaskHandler::stop()
 {
   LOG_INFO("vector kmeans build task start to stop", K_(tg_id));
   if (OB_LIKELY(INVALID_TG_ID != tg_id_)) {
     TG_STOP(tg_id_);
   }
 }
 
 void ObKmeansBuildTaskHandler::wait()
 {
   LOG_INFO("vector kmeans build task handler start to wait", K_(tg_id));
   if (OB_LIKELY(INVALID_TG_ID != tg_id_)) {
     TG_WAIT(tg_id_);
   }
 }
 
 void ObKmeansBuildTaskHandler::destroy()
 {
   LOG_INFO("vector kmeans build task handler start to destroy", K_(tg_id));
   // tg_id is managed by external service, no need to destroy here
   tg_id_ = INVALID_TG_ID;
   is_inited_ = false;
 }
 
 int ObKmeansBuildTaskHandler::get_max_thread_count(int64_t& max_thread_cnt, bool with_refresh /* = false */)
 {
   int ret = OB_SUCCESS;
   if (with_refresh) {
     common::ObSpinLockGuard guard(lock_);
     int64_t tmp_max_thread_cnt = MTL_CPU_COUNT() * THREAD_FACTOR;
     tmp_max_thread_cnt = OB_MAX(tmp_max_thread_cnt, MIN_THREAD_COUNT);
     if (tmp_max_thread_cnt == max_thread_cnt_) {
     } else if (OB_FAIL(TG_SET_ADAPTIVE_THREAD(tg_id_, MIN_THREAD_COUNT,
       tmp_max_thread_cnt))) {
       LOG_WARN("TG_SET_ADAPTIVE_THREAD failed", KR(ret), K_(tg_id));
     } else {
       LOG_INFO("succ to set max thread count", KR(ret), K_(tg_id), K(max_thread_cnt_), K(tmp_max_thread_cnt));
       max_thread_cnt_ = tmp_max_thread_cnt;
     }
   }
   max_thread_cnt = max_thread_cnt_;
   return ret;
 }
 
 int ObKmeansBuildTaskHandler::push_task(ObKmeansBaseTask &task)
 {
   int ret = OB_SUCCESS;
   if (IS_NOT_INIT) {
     ret = OB_NOT_INIT;
     LOG_WARN("handler is not init", KR(ret));
   }
 
   // !!!! inc task ref cnt;
   inc_task_ref();
 
   bool is_push_succ = false;
   int64_t has_retry_cnt = 0;
   while (OB_SUCC(ret) && !is_push_succ && has_retry_cnt++ <= MAX_RETRY_PUSH_TASK_CNT) {
     if (OB_FAIL(TG_PUSH_TASK(tg_id_, &task))) {
       if (ret != OB_EAGAIN) {
         LOG_WARN("fail to TG_PUSH_TASK", KR(ret), K(task));
       } else {
         // sleep 1s and retry
         LOG_INFO("fail to TG_PUSH_TASK, queue is full will retry", KR(ret), K(task));
         ob_usleep(WAIT_RETRY_PUSH_TASK_TIME);
         ret = OB_SUCCESS;
       }
     } else {
       is_push_succ = true;
     }
   }
 
   if (!is_push_succ) {
     if (OB_SUCC(ret)) {
       ret = OB_EAGAIN;
     }
     // !!!!! desc task ref cnt
     dec_task_ref();
     LOG_WARN("fail to push task", KR(ret), K(task), K(is_push_succ));
   }
   return ret;
 }
 
 void ObKmeansBuildTaskHandler::handle(void *task)
 {
   int ret = OB_SUCCESS;
   if (IS_NOT_INIT) {
     ret = OB_NOT_INIT;
     LOG_WARN("handler is not init", KR(ret));
   } else if (OB_ISNULL(task)) {
     ret = OB_INVALID_ARGUMENT;
     LOG_WARN("invalid argument", KR(ret));
   } else {
     ObKmeansBaseTask *base_task = static_cast<ObKmeansBaseTask *>(task);
     if (!base_task->is_finish()) {
       if (OB_FAIL(base_task->do_work())) {
         LOG_WARN("fail to do task", KR(ret));
       }
     }
   }
   // !!!!! desc task ref cnt
   dec_task_ref();
 }
 
 void ObKmeansBuildTaskHandler::handle_drop(void *task)
 {
   int ret = OB_SUCCESS;
   if (IS_NOT_INIT) {
     ret = OB_NOT_INIT;
     LOG_WARN("handler is not init", KR(ret));
   } else if (OB_ISNULL(task)) {
     ret = OB_INVALID_ARGUMENT;
     LOG_WARN("invalid argument", KR(ret));
   } else {
     // thread has set stop.
     // Use base class pointer to handle task
     ObKmeansBaseTask *base_task = static_cast<ObKmeansBaseTask *>(task);
     if (!base_task->is_finish()) {
       base_task->set_finish(OB_CANCELED);
     }
     // !!!!! desc task ref cnt
     dec_task_ref();
   }
 }
 
 /******************************* ObKmeansDistanceCalcTask **********************************/
 int ObKmeansDistanceCalcTask::init(int64_t start_idx, int64_t end_idx,
                                    const ObIArray<float *> *vectors,
                                    float *current_center, float *weight, const int64_t dim)
 {
   int ret = OB_SUCCESS;
   if (OB_UNLIKELY(is_inited_)) {
     ret = OB_INIT_TWICE;
     LOG_WARN("init twice", KR(ret));
   } else if (start_idx < 0 || end_idx < start_idx ||
              OB_ISNULL(vectors) || OB_ISNULL(current_center) ||
              OB_ISNULL(weight) || dim <= 0) {
     ret = OB_INVALID_ARGUMENT;
     LOG_WARN("invalid argument", KR(ret), K(start_idx), K(end_idx),
              KP(vectors), KP(current_center), KP(weight), K(dim));
   } else {
     task_ctx_.vectors_ = vectors;
     task_ctx_.current_center_ = current_center;
     task_ctx_.weight_ = weight;
     task_ctx_.start_idx_ = start_idx;
     task_ctx_.end_idx_ = end_idx;
     task_ctx_.sum_ = 0.0f;
     task_ctx_.dim_ = dim;
     base_ctx_.init();
     is_inited_ = true;
   }
   return ret;
 }
 
 void ObKmeansDistanceCalcTask::reset()
 {
   ObKmeansBaseTask::reset();
   // update ctx
   task_ctx_.reset();
 }
 
 int ObKmeansDistanceCalcTask::do_work()
 {
   int ret = OB_SUCCESS;
   if (!is_inited_ || is_stop()) {
     ret = OB_NOT_INIT;
     LOG_WARN("not init or stop", KR(ret), K_(is_stop));
   } else if (OB_FALSE_IT(base_ctx_.gmt_modified_ = ObTimeUtility::current_time())) {
   } else if (OB_FAIL(ObKmeansAlgo::calc_distances_range(*task_ctx_.vectors_, task_ctx_.start_idx_, task_ctx_.end_idx_,
                                                         task_ctx_.current_center_, task_ctx_.weight_, task_ctx_.dim_,
                                                         task_ctx_.sum_))) {
     LOG_WARN("failed to calc distances range", K(ret));
   }
 
   // update ctx
   set_finish(ret);
   return ret;
 }
 
 /******************************* ObKmeansBuildTask **********************************/
 int ObKmeansBuildTask::init(const common::ObTableID &table_id, const common::ObTabletID &tablet_id, int m_idx,
                             ObKmeansAlgo *algo, const ObIArray<float *> *vectors, ObInsertMonitor *insert_monitor)
 {
   int ret = OB_SUCCESS;
   if (OB_UNLIKELY(is_inited_)) {
     ret = OB_INIT_TWICE;
     LOG_WARN("init twice", KR(ret));
   } else if (m_idx < 0 || OB_ISNULL(algo) || OB_ISNULL(vectors)) {
     ret = OB_INVALID_ARGUMENT;
     LOG_WARN("invalid argument", KR(ret), K(m_idx), KP(algo), KP(vectors));
   } else {
 
     algo_ = algo;
     task_ctx_.vectors_ = vectors;
     task_ctx_.table_id_ = table_id;
     task_ctx_.tablet_id_ = tablet_id;
     task_ctx_.m_idx_ = m_idx;
     task_ctx_.insert_monitor_ = insert_monitor;
     base_ctx_.init();
     is_inited_ = true;
   }
 
   return ret;
 }
 
 int ObKmeansBuildTask::do_work()
 {
   int ret = OB_SUCCESS;
   if (IS_NOT_INIT || is_stop()) {
     ret = OB_NOT_INIT;
     LOG_WARN("not init or stop", KR(ret), K_(is_stop));
   } else if (OB_FALSE_IT(base_ctx_.gmt_modified_ = ObTimeUtility::current_time())) {
   } else if (OB_FAIL(algo_->build(*task_ctx_.vectors_))) {
     LOG_WARN("fail to build", KR(ret), K_(task_ctx), KP_(algo));
   }
   if (OB_NOT_NULL(algo_)) {
       algo_->destroy();
   }
   // update ctx
   base_ctx_.finish(ret);
   if (OB_NOT_NULL(task_ctx_.insert_monitor_) && OB_NOT_NULL(task_ctx_.insert_monitor_->kmeans_monitor_.vec_index_task_finish_cnt_)) {
     (void)ATOMIC_AAF(task_ctx_.insert_monitor_->kmeans_monitor_.vec_index_task_finish_cnt_, 1);
   }
   return ret;
 }
 
 /******************************* ObKmeansAssignTask **********************************/
 int ObKmeansAssignTask::init(int64_t start_idx, int64_t end_idx, ObElkanKmeansAlgo *algo,
                              const ObIArray<float *> *input_vectors, float *centers_distance,
                              int32_t *data_cnt_in_cluster,
                              int32_t *nearest_label_out,
                              float *min_d2_out,
                              const bool accumulate_to_centers,
                              const bool allow_hgraph_assign)
 {
   int ret = OB_SUCCESS;
   const bool require_center_distance =
       (nullptr != algo) ? (!allow_hgraph_assign || !algo->is_hgraph_available()) : true;
   if (OB_ISNULL(algo) || OB_ISNULL(input_vectors) ||
       (accumulate_to_centers && OB_ISNULL(data_cnt_in_cluster)) ||
       (require_center_distance && OB_ISNULL(centers_distance))) {
     ret = OB_INVALID_ARGUMENT;
     SHARE_LOG(WARN, "invalid argument", K(ret), KP(algo), KP(input_vectors), KP(centers_distance),
               KP(data_cnt_in_cluster), K(accumulate_to_centers));
   } else {
     task_ctx_.start_idx_ = start_idx;
     task_ctx_.end_idx_ = end_idx;
     task_ctx_.input_vectors_ = input_vectors;
     task_ctx_.centers_distance_ = centers_distance;
     task_ctx_.data_cnt_in_cluster_ = data_cnt_in_cluster;
     task_ctx_.dis_obj_ = 0.0f;
     task_ctx_.nearest_label_out_ = nearest_label_out;
     task_ctx_.min_d2_out_ = min_d2_out;
     task_ctx_.accumulate_to_centers_ = accumulate_to_centers;
     task_ctx_.allow_hgraph_assign_ = allow_hgraph_assign;
     base_ctx_.init();
     algo_ = algo;
     is_inited_ = true;
   }
   return ret;
 }
 
 void ObKmeansAssignTask::reset()
 {
   ObKmeansBaseTask::reset();
   algo_ = nullptr;
   task_ctx_.reset();
 }
 
 int ObKmeansAssignTask::do_work()
 {
   int ret = OB_SUCCESS;
   if (!is_inited_ || is_stop()) {
     ret = OB_NOT_INIT;
     SHARE_LOG(WARN, "not init or stop", KR(ret), K_(is_stop));
   } else {
     if (OB_FAIL(algo_->assign_vectors_range(*task_ctx_.input_vectors_, task_ctx_.start_idx_, task_ctx_.end_idx_,
                                             task_ctx_.centers_distance_, task_ctx_.data_cnt_in_cluster_,
                                             task_ctx_.dis_obj_, true, task_ctx_.nearest_label_out_,
                                             task_ctx_.min_d2_out_, task_ctx_.accumulate_to_centers_,
                                             task_ctx_.allow_hgraph_assign_))) {
       SHARE_LOG(WARN, "failed to assign vectors range", K(ret));
     }
   }
 
   // update ctx
   set_finish(ret);
   return ret;
 }
 // ------------------ ObKmeansBaseTaskCtx implement ------------------
 void ObKmeansBaseTaskCtx::init()
 {
   gmt_create_ = ObTimeUtility::current_time();
   gmt_modified_ = ObTimeUtility::current_time();
   is_finish_ = false;
   ret_code_ = OB_SUCCESS;
 }
 
 void ObKmeansBaseTaskCtx::finish(int ret_code)
 {
   ATOMIC_STORE(&is_finish_, true);
   ret_code_ = ret_code;
   gmt_modified_ = ObTimeUtility::current_time();
 }
 
 } // end namespace share
 } // end namespace oceanbase
 