package org.butterbrot.floe.distribat;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.media.SoundPool;
import android.os.AsyncTask;
import android.os.Bundle;
import androidx.annotation.NonNull;
import com.google.android.material.floatingactionbutton.FloatingActionButton;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import android.os.Environment;
import android.text.InputType;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.EditText;
import android.widget.ImageView;

import uk.me.berndporr.kiss_fft.KISSFastFourierTransformer;
import org.apache.commons.math3.complex.Complex;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.Arrays;

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
    double[] masterbuf;
    boolean doRecord = false;

    // visualization stuff
    public static int canvas_size = 512;
    ImageView imageView;
    Bitmap bitmap;
    Canvas canvas;
    Paint paint;
    int clearColor = Color.BLACK;

    public static int samplesNumber;

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

    // FIXME: needs to be dynamic for each frequency
    double freq_threshold = 100.0;
    double[] wma = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double[] wmv = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

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
        //basefreq = maxfreq;

        //double maxval = data[basefreq];
        if (maxval < freq_threshold) return 0;

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
    private final String[] permissions = {Manifest.permission.RECORD_AUDIO, Manifest.permission.WRITE_EXTERNAL_STORAGE};
    private static final int REQUEST_PERMISSIONS = 200;

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_PERMISSIONS) {
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

        requestPermissions(permissions, REQUEST_PERMISSIONS);

        showNumberDialog();

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

        masterbuf = new double[samplerate]; // room for one second of data

        FloatingActionButton fab = findViewById(R.id.fab);
        fab.setOnClickListener(view -> {
            if (doRecord) {
                doRecord = false;
                ((FloatingActionButton) view).setImageResource(android.R.drawable.ic_media_play);
            } else {
                ra = (RecordAudioTask) new RecordAudioTask().execute();
                ((FloatingActionButton) view).setImageResource(android.R.drawable.ic_media_pause);
            }
        });

        imageView = this.findViewById(R.id.imageView);
        bitmap = Bitmap.createBitmap(canvas_size, canvas_size, Bitmap.Config.ARGB_8888);
        canvas = new Canvas(bitmap);
        paint = new Paint();
        paint.setColor(Color.GREEN);
        imageView.setImageBitmap(bitmap);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    // https://www.androidcookbook.info/android-media/visualizing-frequencies.html
    private class RecordAudioTask extends AsyncTask<Void, double[], Void> {

        @Override
        protected Void doInBackground(Void... params) {

            int curr = 0;

            while (curr < samplesNumber) {

                doRecord = true;

                try {

                    File file = new File(Environment.getExternalStorageDirectory() + "/AudioRecorder/" + File.separator + System.currentTimeMillis() + ".txt");
                    boolean res = file.createNewFile();

                    FileOutputStream os = null;
                    OutputStreamWriter osw = null;

                    if (file.exists()) {
                        os = new FileOutputStream(file);
                        osw = new OutputStreamWriter(os);
                    }

                    audioRecord.startRecording();
                    Log.d(TAG, "start recording");


                    while (doRecord) {

                        if (count++ % 3 == 0) {
                            nextfreq = (int) (Math.random() * pings.length);
                            soundPool.play(pings[nextfreq], 1.0f, 1.0f, 0, 0, 1.0f);
                        }

                        int result = audioRecord.read(rawbuffer, 0, rawbuffer.length, AudioRecord.READ_BLOCKING);

                        if (checkIfZeros(rawbuffer))
                            continue;

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

                        StringBuilder sb = new StringBuilder();
                        for (float of : outputFloat) {
                            String row = of + "\n";
                            sb.append(row);
                        }

                        Log.v("output", String.valueOf(outputFloat.length));
                        Log.v("output", Arrays.toString(outputFloat));

                        assert osw != null;
                        osw.write(sb.toString());

                        osw.flush();
                        osw.close();
                        os.close();

                        doRecord = false;
                        break;
                    }

                    audioRecord.stop();
                    Log.d(TAG, "stop recording");

                } catch (Exception e) {
                    Log.v(TAG, "An Error Has Occurred" + e);
                }

                curr++;
            }

            doRecord = true;

            runOnUiThread(() -> {
                FloatingActionButton fab = findViewById(R.id.fab);
                fab.performClick();
                showNumberDialog();
            });

            return null;
        }

        private boolean checkIfZeros(short[] arr) {
            for (short value : arr) {
                if (value != 0)
                    return false;
            }
            return true;
        }

        // https://stackoverflow.com/questions/5511250/capturing-sound-for-analysis-and-visualizing-frequencies-in-android
        @Override
        protected void onProgressUpdate(double[]... data) {
            canvas.drawColor(clearColor);
            clearColor = Color.BLACK;
            for (int x = 0; x < canvas_size; x++) {
                // visualize only the uppermost part of the spectrum
                int startbin = (data[0].length / 2) - canvas_size;
                int y1 = (int) (canvas_size - (data[0][startbin + x] * 10));
                int y2 = canvas_size;
                canvas.drawLine(x, y1, x, y2, paint);
            }
            imageView.invalidate();
        }
    }

    @SuppressLint("RestrictedApi")
    private void showNumberDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
        builder.setTitle("How many samples do you want to record?");

        // Set up the input
        final EditText input = new EditText(MainActivity.this);

        // Specify the type of input expected; this, for example, sets the input as a password, and will mask the text
        input.setInputType(InputType.TYPE_CLASS_NUMBER);
        input.setRawInputType(Configuration.KEYBOARD_12KEY);
        builder.setView(input, 50, 0, 50, 0);

        // Set up the buttons
        builder.setPositiveButton("OK", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                samplesNumber = Integer.parseInt(input.getText().toString());
            }
        });
        builder.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                dialog.cancel();
            }
        });

        builder.show();
    }
}