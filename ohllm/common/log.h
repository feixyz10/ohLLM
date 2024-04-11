#pragma once

#include <stdlib.h>

typedef enum {
  LOG_LEVEL_DEBUG = 0,
  LOG_LEVEL_INFO,
  LOG_LEVEL_WARN,
  LOG_LEVEL_ERROR,
  LOG_LEVEL_FATAL,
  LOG_LEVEL_NUMBER
} LOG_LEVEL;

namespace internal {

void _Log(LOG_LEVEL level, const char *const filename, int line,
          const char *const format, ...);

void _DoNothing(LOG_LEVEL level, const char *const format, ...);

}  // namespace internal

#ifdef BUILD_LOG

// debug: cyan
#define LOGD(format, ...) \
  ::internal::_Log(LOG_LEVEL_DEBUG, __FILE__, __LINE__, format, ##__VA_ARGS__)

// info: white
#define LOGI(format, ...) \
  ::internal::_Log(LOG_LEVEL_INFO, __FILE__, __LINE__, format, ##__VA_ARGS__)

// warning: yellow
#define LOGW(format, ...) \
  ::internal::_Log(LOG_LEVEL_WARN, __FILE__, __LINE__, format, ##__VA_ARGS__)

// error: red
#define LOGE(format, ...) \
  ::internal::_Log(LOG_LEVEL_ERROR, __FILE__, __LINE__, format, ##__VA_ARGS__)

// fatal: dark red
#define LOGF(format, ...) \
  ::internal::_Log(LOG_LEVEL_FATAL, __FILE__, __LINE__, format, ##__VA_ARGS__)

#else

#define LOGD(format, ...) \
  ::internal::_DoNothing(LOG_LEVEL_DEBUG, format, ##__VA_ARGS__)
#define LOGI(format, ...) \
  ::internal::_DoNothing(LOG_LEVEL_INFO, format, ##__VA_ARGS__)
#define LOGW(format, ...) \
  ::internal::_DoNothing(LOG_LEVEL_WARN, format, ##__VA_ARGS__)
#define LOGE(format, ...) \
  ::internal::_DoNothing(LOG_LEVEL_ERROR, format, ##__VA_ARGS__)
#define LOGF(format, ...) \
  ::internal::_DoNothing(LOG_LEVEL_FATAL, format, ##__VA_ARGS__)

#endif  // BUILD_LOG


#define LOGD_IF(cond, format, ...) \
  if (cond) LOGD(format, ##__VA_ARGS__)
#define LOGI_IF(cond, format, ...) \
  if (cond) LOGI(format, ##__VA_ARGS__)
#define LOGW_IF(cond, format, ...) \
  if (cond) LOGW(format, ##__VA_ARGS__)
#define LOGE_IF(cond, format, ...) \
  if (cond) LOGE(format, ##__VA_ARGS__)
#define LOGF_IF(cond, format, ...) \
  if (cond) LOGF(format, ##__VA_ARGS__)
#define ASSERT(cond, format, ...) \
  if (!(cond)) LOGF(format, ##__VA_ARGS__)
