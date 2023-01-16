package org.butterbrot.floe.distribat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.media.SoundPool;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.app.NotificationCompat;
import androidx.core.app.NotificationManagerCompat;

import com.google.android.material.floatingactionbutton.FloatingActionButton;

import org.apache.commons.math3.complex.Complex;
import org.butterbrot.floe.distribat.ml.HeadsUpModel;
import org.butterbrot.floe.distribat.ml.HeadsupMetadata2;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.util.Arrays;

import uk.me.berndporr.kiss_fft.KISSFastFourierTransformer;

public class MainActivity extends AppCompatActivity {

    public static String TAG = "DistriBat";

    // frequency resolution is ~  5.86 Hz with 8192
    // frequency resolution is ~ 11.71 Hz with 4096
    // frequency resolution is ~ 46.87 Hz with 1024
    // processing takes ~ 15 ms
    public static int samplerate = 48000;
    public static int fftwindowsize = 4096;
    public static int searchwindow = 25;

    public RecordAudioTask ra;
    public KISSFastFourierTransformer fft;
    public AudioRecord audioRecord;

    SoundPool soundPool;
    int[] pings;
    public int count = 0;
    public int nextfreq = 0;

    short[] rawbuffer;
    double[] input;
    double[] prev;
    double[] hann;
    double[] scratch;
    boolean doRecord = false;

    int obstacleCnt = 0;
    boolean previous = false;

    private int ComputeIndex(int frequency) {
        return (int) ((((double) fftwindowsize) / ((double) samplerate)) * (double) frequency);
    }

    // Hann window is generally considered the best all-purpose window function, see also ...
    // https://dsp.stackexchange.com/questions/22175/how-should-i-select-window-size-and-overlap
    private double[] hann_window(int size) {
        double[] hann = new double[size];
        for (int i = 0; i < size; i++) {
            hann[i] = 0.5 * (1.0 - Math.cos(2.0 * i * Math.PI / (double) (size - 1)));
        }
        return hann;
    }

    double[] fft_with_hann(double[] input, int offset) {
        for (int i = 0; i < scratch.length; i++)
            scratch[i] = hann[i] * input[i + offset];
        Complex[] tmp = fft.transformRealOptimisedForward(scratch);
        for (int i = 0; i < tmp.length; i++)
            scratch[i] = tmp[i].abs();
        return scratch;
    }

    int[] freq_offsets = {
            19500,
            19700,
            19900,
            20100,
            20300,
            20500
    };

    double[] template = new double[2 * searchwindow + 1];

    double freq_threshold = 100.0;

    // detect "interesting" frequencies in FFT result
    private int detect_freq(double[] data) {
        // first run -> convert frequencies to FFT bins
        if (freq_offsets[0] > 10000)
            for (int i = 0; i < freq_offsets.length; i++) {
                freq_offsets[i] = ComputeIndex(freq_offsets[i]);
                Log.d(TAG, "mapping index " + freq_offsets[i]);
            }

        // find local maximum around expected base frequency
        //int basefreq = freq_offsets[nextfreq];
        double maxval = 0.0;
        int basefreq = 0;
        for (int i = 0; i < freq_offsets.length; i++) { // = basefreq-searchwindow; i < basefreq+searchwindow; i++) {
            int index = freq_offsets[i];
            if (data[index] > maxval) {
                maxval = data[index];
                basefreq = index;
                nextfreq = i;
            }
        }

        if (maxval < freq_threshold)
            return 0;

        StringBuilder msg = new StringBuilder("values: ");

        for (int i = 0; i <= 2 * searchwindow; i++) {
            template[i] = 0.9 * template[i] + 0.1 * (data[basefreq - searchwindow + i] / maxval);
            msg.append((int) (100 * data[basefreq - searchwindow + i] / maxval)).append(",");
            data[basefreq - searchwindow + i] -= maxval * template[i];
        }

        Log.d(TAG, msg.toString());
        msg = new StringBuilder("template: ");
        for (double t : template) msg.append((int) (t * 100)).append(",");
        Log.d(TAG, msg.toString());

        // find weight distribution of peak
        double balance = 0.0;
        for (int i = 5; i <= searchwindow; i++)
            balance += (data[basefreq + i] - data[basefreq - i]);

        Log.v(TAG, "basefreq = " + basefreq + " balance = " + balance);
        return (int) Math.round(balance);
    }

    // Requesting permission to RECORD_AUDIO (from https://developer.android.com/guide/topics/media/mediarecorder#java)
    private boolean permissionToRecordAccepted = false;
    private final String[] permissions = {Manifest.permission.RECORD_AUDIO};
    private static final int REQUEST_RECORD_AUDIO_PERMISSION = 200;

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_RECORD_AUDIO_PERMISSION) {
            permissionToRecordAccepted = (grantResults[0] == PackageManager.PERMISSION_GRANTED);
        }
        if (!permissionToRecordAccepted) finish();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        requestPermissions(permissions, REQUEST_RECORD_AUDIO_PERMISSION);

        soundPool = new SoundPool.Builder().build();
        pings = new int[6];
        pings[0] = soundPool.load(this, R.raw.sine19500, 0);
        pings[1] = soundPool.load(this, R.raw.sine19700, 0);
        pings[2] = soundPool.load(this, R.raw.sine19900, 0);
        pings[3] = soundPool.load(this, R.raw.sine20100, 0);
        pings[4] = soundPool.load(this, R.raw.sine20300, 0);
        pings[5] = soundPool.load(this, R.raw.sine20500, 0);

        fft = new KISSFastFourierTransformer();

        int bufferSize = AudioRecord.getMinBufferSize(samplerate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        Log.d(TAG, "minBufferSize = " + bufferSize);
        audioRecord = new AudioRecord(MediaRecorder.AudioSource.UNPROCESSED, samplerate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufferSize * 4);

        rawbuffer = new short[fftwindowsize];
        input = new double[fftwindowsize];
        prev = new double[fftwindowsize];
        hann = hann_window(fftwindowsize);
        scratch = new double[fftwindowsize];

        FloatingActionButton fab = findViewById(R.id.fab);
        fab.setOnClickListener(view -> {
            TextView textView = findViewById(R.id.statusText);
            if (doRecord) {
                doRecord = false;
                ((FloatingActionButton) view).setImageResource(android.R.drawable.ic_media_play);
                textView.setText(R.string.click_play_to_start_recording);
            } else {
                ra = (RecordAudioTask) new RecordAudioTask().execute();
                ((FloatingActionButton) view).setImageResource(android.R.drawable.ic_media_pause);
                textView.setText(R.string.recording_started);
            }
        });
    }

    // https://www.androidcookbook.info/android-media/visualizing-frequencies.html
    private class RecordAudioTask extends AsyncTask<Void, double[], Void> {

        @Override
        protected Void doInBackground(Void... params) {

            doRecord = true;
            HeadsupMetadata2 model = null;
            try {
                 model = HeadsupMetadata2.newInstance(MainActivity.this);
            } catch (IOException e) {
                e.printStackTrace();
            }

            // Creates inputs for reference.
            TensorBuffer audioClip = TensorBuffer.createFixedSize(new int[]{4096}, DataType.FLOAT32);

            try {

                audioRecord.startRecording();
                Log.d(TAG, "start recording");

                while (doRecord) {

                    if (count++ % 3 == 0) {
                        nextfreq = (int) (Math.random() * pings.length);
                        soundPool.play(pings[nextfreq], 1.0f, 1.0f, 0, 0, 1.0f);
                    }

                    int result = audioRecord.read(rawbuffer, 0, rawbuffer.length, AudioRecord.READ_BLOCKING);

                    double[] tmpb = prev;
                    prev = input;
                    input = tmpb;

                    for (int i = 0; i < input.length; i++)
                        input[i] = 100.0 * (rawbuffer[i] / (double) Short.MAX_VALUE);
                    double[] output = fft_with_hann(input, 0);

                    float[] outputFloat = new float[output.length];
                    for (int i = 0; i < output.length; i++) {
                        outputFloat[i] = (float) output[i];
                    }

                    Log.v("output length", String.valueOf(outputFloat.length));
                    Log.v("output", Arrays.toString(outputFloat));

                    audioClip.loadArray(outputFloat);

                    // Runs model inference and gets result.
                    assert model != null;
                    HeadsupMetadata2.Outputs outputs = model.process(audioClip);
                    TensorBuffer probability = outputs.getProbabilityAsTensorBuffer();

                    Log.v("Probability", Arrays.toString(probability.getFloatArray()));

                    float probabilityRounded = probability.getFloatArray()[0];

                    runOnUiThread(() -> {

                        final ConstraintLayout resultsLayout = findViewById(R.id.resultsLayout);
                        final TextView results = findViewById(R.id.resultsText);

                        if (probabilityRounded > 0.5 && obstacleCnt >= 2) {

                            obstacleCnt++;

                            NotificationCompat.Builder builder = new NotificationCompat.Builder(MainActivity.this,
                                    "My Notification");

                            builder.setContentTitle("STOP!!!")
                                    .setContentText("The wall is in front of you!")
                                    .setSmallIcon(R.drawable.app_logo)
                                    .setAutoCancel(true);

                            NotificationManagerCompat manager = NotificationManagerCompat.from(MainActivity.this);
                            manager.notify(1, builder.build());


                            String status = "Watch Your Head!\nProbability of an Obstacle: " + probabilityRounded * 100 + "%";
                            results.setText(status);
                            results.setTextColor(Color.RED);
                        } else {
                            if (probabilityRounded > 0.5 && previous) {
                                obstacleCnt++;
                            } else if (probabilityRounded > 0.5 && !previous) {
                                obstacleCnt = 0;
                                previous = true;
                            } else {
                                obstacleCnt = 0;
                                previous = false;
                            }

                            String status = "Walk...\nProbability of an Obstacle: " + probabilityRounded * 100 + "%";
                            results.setText(status);
                            results.setTextColor(Color.GREEN);
                        }
                        resultsLayout.setVisibility(View.VISIBLE);
                    });
                }

                // Releases model resources if no longer used.
                assert model != null;
                model.close();

                audioRecord.stop();
                Log.d(TAG, "stop recording");

            } catch (Exception e) {
                e.printStackTrace();
            }

            return null;
        }
    }
}