#include "log.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "macro.h"

namespace internal {

void _Time2String(char *time_string) {
  time_t t = time(NULL);
  CLOCKS_PER_SEC;
  struct tm *tm_info = localtime(&t);
  strftime(time_string, 26, "%Y%m%d_%H:%M:%S", tm_info);
}

void _Log(LOG_LEVEL level, const char *const filename, int line,
          const char *const format, ...) {
  const int colors[] = {36, 37, 33, 31, 31};
  const char levels[] = "DIWEF";
  char time_string[26] = {0};
  int bold = level != LOG_LEVEL_FATAL ? 0 : 1;
  _Time2String(time_string);
  printf("\033[%01d;%02dm[%c %s %s:%d] ", bold, colors[level], levels[level],
         time_string, filename, line);
  va_list ap;
  va_start(ap, format);
  vprintf(format, ap);
  va_end(ap);
  printf("\033[0m\n");
  if (level == LOG_LEVEL_FATAL) { exit(1); }
}

void _DoNothing(LOG_LEVEL level, const char *const format, ...) {
  UNUSED(format);
  if (level == LOG_LEVEL_FATAL) { exit(1); }
}

}  // namespace internal