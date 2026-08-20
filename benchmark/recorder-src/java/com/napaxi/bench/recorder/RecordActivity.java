package com.napaxi.bench.recorder;

import android.app.Activity;
import android.content.Intent;
import android.media.projection.MediaProjectionManager;
import android.os.Bundle;

/** Headless launcher: asks for the media-projection grant, then forwards the
 * result to RecordService with the requested output path. */
public class RecordActivity extends Activity {
    private static final int REQUEST = 4711;
    private String outPath;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        outPath = getIntent().getStringExtra("out_path");
        if (outPath == null) outPath = "/sdcard/bench.mp4";
        MediaProjectionManager manager =
                (MediaProjectionManager) getSystemService(MEDIA_PROJECTION_SERVICE);
        // Android 14+ requires FOREGROUND_SERVICE_MEDIA_PROJECTION before the
        // grant dialog; starting the service only after user approval below.
        startActivityForResult(manager.createScreenCaptureIntent(), REQUEST);
    }

    @Override
    @SuppressWarnings("deprecation")
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != REQUEST) { finish(); return; }
        if (resultCode != RESULT_OK || data == null) { finish(); return; }
        Intent service = new Intent(this, RecordService.class)
                .putExtra("result_code", resultCode)
                .putExtra("data", data)
                .putExtra("out_path", outPath);
        startForegroundService(service);
        finish();
    }
}
