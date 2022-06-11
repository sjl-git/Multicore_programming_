#ifndef _ERROR_H_
#define _ERROR_H_


#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/prctl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <string>
#include <sstream>
#include <iostream>
#include <assert.h>



#define ASSERT(left,operator,right) { if(!((left) operator (right))){ std::cerr << "ASSERT FAILED: " << #left << #operator << #right << " @ " << __FILE__ << " (" << __LINE__ << "). " << #left << "=" << (left) << "; " << #right << "=" << (right) << std::endl; assert((left) operator (right));}}

 void format_backtrace(std::string &output) {
  // Trim the right side of the output.
  size_t index = output.find_last_not_of(" \t\r\n");
  output.erase((index != std::string::npos) ? index + 1 : 0);

  // Find an entry for the signal handler and trim all entries deeper than it.
  index = output.find("<signal handler called>");
  if (index != std::string::npos) {
    index = output.find('\n', index);
    if (index != std::string::npos) {
      output.erase(0, index + 1);
    }
  }

  // Insert indentation.
  if (!output.empty()) {
    const std::string indent = "  ";
    output.insert(0, indent);

    for (size_t offset = output.find('\n'); offset != std::string::npos; ) {
      output.insert(offset + 1, indent);
      offset = output.find('\n', offset + 1);
    }
  }
}
void print_backtrace() {
  int pipefd[2];
  int use_pipe = (pipe(pipefd) == 0);

  pid_t child = fork();
  if (child == -1) {
    return;
  }

  if (child == 0) {
    if (use_pipe) {
      close(pipefd[0]);
    }

    // Create a dummy process so that GDB can attach to it. This is to provide a
    // workaround for the case when ptrace_scope is enabled, in which processes
    // are not allowed to attach to non-child processes.
    pid_t dummy = fork();
    if (dummy == -1) {
      _Exit(1);
    }

    if (dummy == 0) {
      if (use_pipe) {
        close(pipefd[1]);
      }

      // Wait until the parent process sends SIGTERM on its termination.
      int result = prctl(PR_SET_PDEATHSIG, SIGTERM, 0, 0, 0);
      if (result == 0) {
        pause();
      }
    } else {
      // Ignore stderr and redirect stdout to pipe (or stderr).
      dup2(use_pipe ? pipefd[1] : fileno(stderr), fileno(stdout));
      //FILE* tmp = freopen("/dev/null", "w", stderr);

      // Run GDB to print the current stack trace.
      std::string pid = std::to_string(dummy);
      execlp("gdb", "gdb", "--pid", pid.c_str(), "--batch",
             "-ex", "set print frame-arguments none", "-ex", "bt", NULL);
    }

    _Exit(0);
  } else {
    if (use_pipe) {
      close(pipefd[1]);

      // Read backtrace from GDB.
      std::string output;
      for (char c; read(pipefd[0], &c, 1) > 0; ) {
        output.push_back(c);
      }

      close(pipefd[0]);

      // Format the output.
      format_backtrace(output);

      // Print the backtrace.
      if (!output.empty()) {
        fprintf(stderr, "Backtrace:\n%s\n", output.c_str());
      }
    }

    waitpid(child, NULL, 0);
  }
}


extern "C" void backtrace_handler(int sig) {
  // Print the current stack trace.
  print_backtrace();

  // Call the original signal handler.
  signal(sig, SIG_DFL);
  int result = raise(sig);

  if (result != 0) {
    _Exit(sig);
  }
}

void install_backtrace_handler() {
  signal(SIGABRT, backtrace_handler);
  signal(SIGFPE, backtrace_handler);
  signal(SIGSEGV, backtrace_handler);
}
#endif 
