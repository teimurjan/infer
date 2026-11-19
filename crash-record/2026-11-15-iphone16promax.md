# Crash record: iPhone 16 Pro Max, 2026-11-15

Device: iPhone 16 Pro Max (iPhone17,2), iOS 26.7.1 (23H30), Safari 26.6.2. Times are phone local time (+06:00).

Sources: the phone log archive (`log collect` from 16:45), and the crash reports copied with `devicectl` (`systemCrashLogs`). The lines and reports are in `evidence/`.

The exit tuple is `(namespace, code, signal)` from launchd. Namespace 1 is jetsam, the iOS memory killer. Code 7 is the per-process memory limit. The jetsam report for PID 36687 confirms this code. Signal 9 is SIGKILL.

## Memory probe kills

| Ramp | Last good step | Time | Process | Reason | Source |
|---|---|---|---|---|---|
| wasm | 1920 MB | 17:30:06.768 | WebContent 36687 | jetsam `per-process-limit` at 2048 MB, exit (1, 7, 9) | `JetsamEvent-2026-11-15-173007.ips`, launchd exit line |
| js | 1984 MB | 17:30:08.668 | WebContent 36688 | exit (1, 7, 9), per-process limit | launchd exit line |
| gpu | 1920 MB | 17:30:18.177 | WebContent 36730 | exit (1, 7, 9), per-process limit | launchd exit line |
| mixed | 1920 MB | 17:30:19.797 | WebContent 36733 | exit (1, 7, 9), per-process limit | launchd exit line |

Safari logged each of the four as `WebProcessProxy::processDidTerminateOrFailedToLaunch: reason=Crash`. Safari uses this label for a process that the system kills.

iOS wrote a jetsam report for the first kill only. The other three kills appear only in the log.

The per-tab limit is 2048 MB (131072 pages of 16 KB). The GPU ramp died at the same limit as the CPU ramps. WebKit charges the tab's WebGPU buffers to the tab process, so GPU memory counts against the same 2 GB.

The mapping of ramp to PID uses order. The probe runs the ramps in the order wasm, js, gpu, mixed, and the four kills happened in that order. The breadcrumbs from this run have no clock time, so the times do not confirm the mapping directly.

Two other Safari processes ended in the same window. They were not probe tabs.

| Time | Process | Reason |
|---|---|---|
| 17:30:06.164 | WebContent 36437 | Safari asked RunningBoard to end it, exit (1, 17, 9). Code 17 is not decoded. |
| 17:30:09.415 | WebContent 36401 | Background tab (151 MB, suspended), exit (1, 9, 9), jetsam idle exit |

## Watchdog probe kills

| Single submit | Time | Process | Reason | Source |
|---|---|---|---|---|
| ~800 ms | 22:50:49.647 | com.apple.WebKit.GPU 38870 | `GPU Hang Error (kIOGPUCommandBufferCallbackErrorHang)`, firmware `restart_reason 4: firmware-detected lockup` | log line, `gpuEvent-...-225049.ips` |
| ~566 ms | 22:50:50.164 | com.apple.WebKit.GPU 38870 | same | log line, `gpuEvent-...-225050.ips` |
| ~519 ms | 22:50:51.168 | com.apple.WebKit.GPU 38870 | same | log line, `gpuEvent-...-225051.ips` |

The gaps between the three hang errors are 0.52 s and 1.00 s. The gaps between the three losses in the breadcrumbs are 0.51 s and 1.01 s.

The GPU process survived all three hangs, because it has the same PID each time. The GPU firmware reset the work, and WebKit reported a lost device to the page. The tab did not die.

## Other GPU errors in the log

At 19:51:24 and 21:18:57, a WebKit GPU process logged `kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted`. These GPU processes belonged to the eBay app (PID 37746) and the Amazon app (PID 38273), not to Safari. They are not related to the probes. They show that iOS refuses GPU work from an app in the background. If you leave Safari during a run, expect a lost device for this reason.

## SD-Turbo baseline

| Stage | Time | Process | Reason | Source |
|---|---|---|---|---|
| unet fetch | 23:20:43.790 | WebContent 39079 | exit (1, 7, 9), per-process limit | launchd exit line, `evidence/sd-turbo-baseline-log-excerpt.txt` |

The page started at 23:17:18. Its last breadcrumb was `unet fetch start`, so the tab died during the 1.7 GB unet download, before session creation. Safari logged `reason=Crash` and reloaded the page at 23:20:44.

The GPU process did not die. The log has no GPU process exit and no Metal error in this window. iOS wrote no jetsam report for this kill.

## How to repeat this

1. Run the probe on the phone, with the phone connected to the Mac.
2. Copy the crash reports: `xcrun devicectl device copy from --device <UDID> --domain-type systemCrashLogs --source / --destination <dir>`.
3. Collect the log in a terminal, because `sudo` asks for a password: `sudo /usr/bin/log collect --device-udid <UDID> --start '<time before the run>' --output <dir>/phone.logarchive`.
4. Find the exits: `log show <archive> --predicate 'process == "runningboardd" AND eventMessage CONTAINS "termination reported by launchd"'`.
5. Find GPU hangs: `log show <archive> --predicate 'eventMessage CONTAINS "kIOGPUCommandBufferCallbackError"'`.

The breadcrumb server now adds a clock time (`at`) to each line. In the next run, you can match each kill to its log line by time, not only by order.
