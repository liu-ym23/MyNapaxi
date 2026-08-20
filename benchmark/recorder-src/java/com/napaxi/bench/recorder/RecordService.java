package com.napaxi.bench.recorder;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.MediaRecorder;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.IBinder;
import java.io.File;

/** Foreground service mirroring the default display into an H264 mp4 until
 * the harness delivers the com.napaxi.bench.recorder.STOP action (which
 * finalizes the container via onDestroy) or force-stops the process. */
public class RecordService extends Service {
    private MediaProjection projection;
    private VirtualDisplay display;
    private MediaRecorder recorder;

    @Override
    public IBinder onBind(Intent intent) { return null; }

    @Override
    @SuppressWarnings({"deprecation", "UnspecifiedRegisterReceiverFlag"})
    public int onStartCommand(Intent intent, int flags, int startId) {
        // Explicit stop action: finalize the mp4 (MediaRecorder.stop() writes
        // the moov index) and shut down. `am stopservice` cannot stop a
        // media-projection foreground service on some ROMs, and force-stop
        // skips onDestroy entirely — both leave the container unplayable.
        if (intent != null && "com.napaxi.bench.recorder.STOP".equals(intent.getAction())) {
            stopSelf();
            return START_NOT_STICKY;
        }
        startInForeground();
        int resultCode = intent != null ? intent.getIntExtra("result_code", 0) : 0;
        Intent data = intent != null ? intent.getParcelableExtra("data") : null;
        String outPath = intent != null ? intent.getStringExtra("out_path") : "/sdcard/bench.mp4";
        if (data == null) { stopSelf(); return START_NOT_STICKY; }

        MediaProjectionManager manager =
                (MediaProjectionManager) getSystemService(MEDIA_PROJECTION_SERVICE);
        projection = manager.getMediaProjection(resultCode, data);

        File out = new File(outPath);
        // /sdcard legacy paths are not writable by MediaRecorder's native
        // layer for targetSdk 29+. Route recordings through the app's own
        // external files dir and expose the final path via the notification
        // / logcat; the harness pulls from getExternalFilesDir.
        {
            File fallback = new File(getExternalFilesDir(null), out.getName());
            out = fallback;
            outPath = fallback.getAbsolutePath();
        }
        if (out.getParentFile() != null) out.getParentFile().mkdirs();
        recorder = Build.VERSION.SDK_INT >= 31
                ? new MediaRecorder(this)
                : new MediaRecorder();
        int width = getResources().getDisplayMetrics().widthPixels;
        int height = getResources().getDisplayMetrics().heightPixels;
        int dpi = getResources().getDisplayMetrics().densityDpi;
        // Cap to ~720p and align to 16: native phone resolutions can exceed
        // the encoder's capability and make prepare() fail with -2147483648.
        int maxWidth = 480;
        if (width > maxWidth) {
            height = height * maxWidth / width;
            width = maxWidth;
        }
        width = width / 16 * 16;
        height = height / 16 * 16;
        recorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
        recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        recorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
        recorder.setVideoSize(width, height);
        recorder.setVideoFrameRate(15);
        recorder.setVideoEncodingBitRate(6_000_000);
        android.util.Log.i("BenchRecorder", "recording to " + outPath);
        recorder.setOutputFile(outPath);
        try {
            recorder.prepare();
        } catch (Exception e) {
            stopSelf();
            return START_NOT_STICKY;
        }
        display = projection.createVirtualDisplay("bench", width, height, dpi,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                recorder.getSurface(), null, null);
        recorder.start();
        return START_NOT_STICKY;
    }

    private void startInForeground() {
        NotificationManager nm = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
        NotificationChannel channel = new NotificationChannel(
                "bench_recorder", "Benchmark Recorder", NotificationManager.IMPORTANCE_LOW);
        nm.createNotificationChannel(channel);
        Notification notification = new Notification.Builder(this, "bench_recorder")
                .setContentTitle("Benchmark 录屏中")
                .setSmallIcon(android.R.drawable.ic_media_play)
                .build();
        if (Build.VERSION.SDK_INT >= 29) {
            startForeground(1, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);
        } else {
            startForeground(1, notification);
        }
    }

    @Override
    public void onDestroy() {
        try { if (recorder != null) recorder.stop(); } catch (Exception ignored) {}
        if (recorder != null) recorder.release();
        if (display != null) display.release();
        if (projection != null) projection.stop();
        super.onDestroy();
    }
}
