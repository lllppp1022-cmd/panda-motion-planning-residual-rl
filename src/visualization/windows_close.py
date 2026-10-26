"""Route our own Windows viewer's close request through the command queue.

MuJoCo 3.16.0 native close can destroy the C++ mutex while a passive physics
thread is entering Handle.lock (reproduced access violation on this machine).
Do not change MuJoCo or global windows. Subclass only this process's one named
GLFW window. WM_CLOSE becomes queued intent; the physics owner calls close()
after leaving its loop/locks. All other messages go to the original procedure.
"""
import ctypes
import os
import sys
import time

# Keep WNDPROC alive until interpreter exit, including late WM_NCDESTROY.
_live_hooks = []


class QueuedWindowClose:
    def __init__(self, keyboard, title):
        self.hwnd = None
        self.requests = 0
        if sys.platform != 'win32':
            return
        from ctypes import wintypes as w
        self.user = ctypes.WinDLL('user32', use_last_error=True)
        self.user.GetWindowThreadProcessId.argtypes = [w.HWND, ctypes.POINTER(w.DWORD)]
        self.user.GetWindowTextW.argtypes = [w.HWND, w.LPWSTR, ctypes.c_int]
        self.user.GetWindowTextW.restype = ctypes.c_int
        self.user.IsWindow.argtypes = [w.HWND]
        self.user.IsWindow.restype = w.BOOL
        self.user.SetWindowLongPtrW.argtypes = [w.HWND, ctypes.c_int, ctypes.c_ssize_t]
        self.user.SetWindowLongPtrW.restype = ctypes.c_ssize_t
        self.user.CallWindowProcW.argtypes = [ctypes.c_void_p, w.HWND, w.UINT, w.WPARAM, w.LPARAM]
        self.user.CallWindowProcW.restype = ctypes.c_ssize_t
        enum_proc = ctypes.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
        self.user.EnumWindows.argtypes = [enum_proc, w.LPARAM]
        found = []

        @enum_proc
        def collect(hwnd, _):
            pid = w.DWORD()
            self.user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            name = ctypes.create_unicode_buffer(512)
            self.user.GetWindowTextW(hwnd, name, len(name))
            if pid.value == os.getpid() and name.value == title:
                found.append(hwnd)
            return True

        self.user.EnumWindows(collect, 0)
        if len(found) != 1:
            raise RuntimeError(f'Expected own named viewer window, found {len(found)}')
        self.hwnd = found[0]
        wndproc = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, w.HWND, w.UINT, w.WPARAM, w.LPARAM)

        @wndproc
        def procedure(hwnd, message, wp, lp):
            if message == 0x0010:  # WM_CLOSE: no physics/viewer calls in callback.
                keyboard.callback(256)
                self.requests += 1
                return 0
            return self.user.CallWindowProcW(ctypes.c_void_p(self.original), hwnd, message, wp, lp)

        self.procedure = procedure
        ctypes.set_last_error(0)
        self.original = self.user.SetWindowLongPtrW(self.hwnd, -4, ctypes.cast(procedure, ctypes.c_void_p).value)
        if not self.original:
            raise ctypes.WinError(ctypes.get_last_error())
        _live_hooks.append(self)

    def alive(self):
        return bool(self.hwnd and self.user.IsWindow(self.hwnd))

    def wait_destroyed(self, seconds=5):
        deadline = time.perf_counter() + seconds
        while self.alive() and time.perf_counter() < deadline:
            time.sleep(.01)
        return not self.alive()
