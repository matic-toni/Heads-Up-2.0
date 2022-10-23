package org.butterbrot.floe.distribat;

import android.Manifest;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
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
import android.os.Environment;
import android.support.annotation.NonNull;
import android.support.design.widget.FloatingActionButton;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.ImageView;
import android.widget.Toast;

import uk.me.berndporr.kiss_fft.KISSFastFourierTransformer;
import org.apache.commons.math3.complex.Complex;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    public static String TAG = "DistriBat";

    private static final String AUDIO_RECORDER_FOLDER = "AudioRecorder";
    private static final String AUDIO_RECORDER_EXT_FILE = ".wav";
    private static final String AUDIO_RECORDER_TEMP_FILE = "record_temp.raw";
    private static final int RECORDER_BPP = 16;

    private static String isPositive;
    private static final int numberOfIterations = 1;

    // frequency resolution is ~  5.86 Hz with 8192
    // frequency resolution is ~ 11.71 Hz with 4096
    // frequency resolution is ~ 46.87 Hz with 1024
    // processing takes ~ 15 ms
    public static int SAMPLE_RATE = 48000;
    public static int fftwindowsize = 4096;
    public static int searchwindow = 25;

    public RecordAudioTask ra;
    public KISSFastFourierTransformer fft;
    public AudioRecord audioRecord;

    // frequencies
    SoundPool soundPool;
    int[] pings;
    public int count = 0;
    public int nextfreq = 0;
    public int eventcount = 0;

    short[] rawbuffer;
    double[] input;
    double[] prev;
    double[] hann;
    double[] scratch;
    double[] masterbuf;
    double[] noisefloor;
    boolean doRecord = false;
    int master_offset = 0;

    // visualization stuff
    public static int canvas_size = 512;
    ImageView imageView;
    Bitmap bitmap;
    Canvas canvas;
    Paint paint;
    int clearColor = Color.BLACK;


    // https://stackoverflow.com/questions/5774104/android-audio-fft-to-retrieve-specific-frequency-magnitude-using-audiorecord
    private double ComputeFrequency(int arrayIndex) {
        return ((1.0 * SAMPLE_RATE) / (1.0 * fftwindowsize)) * arrayIndex;
    }

    private int ComputeIndex(int frequency) {
        return (int)((((double)fftwindowsize) / ((double)SAMPLE_RATE)) * (double)frequency);
    }

    // Hann window is generally considered the best all-purpose window function, see also ...
    // https://dsp.stackexchange.com/questions/22175/how-should-i-select-window-size-and-overlap
    private double[] hann_window(int size) {
        double[] hann = new double[size];
        for (int i = 0; i < size; i++) {
            hann[i] = 0.5 * (1.0 - Math.cos(2.0*i*Math.PI/(double)(size-1)));
        }
        return hann;
    }

    double[] fft_with_hann(double[] input, int offset) {
        for (int i = 0; i < scratch.length; i++) scratch[i] = hann[i] * input[i+offset];
        Complex[] tmp = fft.transformRealOptimisedForward(scratch);
        for (int i = 0; i < tmp.length; i++) scratch[i] = tmp[i].abs();
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

    double[] template = new double[2*searchwindow + 1];

    // FIXME: needs to be dynamic for each frequency
    double freq_threshold = 100.0;
    // w moving average, variance
    double[] wma = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    double[] wmv = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    // detect "interesting" frequencies in FFT result
    private int detect_freq(double[] data) {
        // first run -> convert frequencies to FFT bins
        if (freq_offsets[0] > 10000)
            for (int i = 0; i < freq_offsets.length; i++) {
                freq_offsets[i] = ComputeIndex(freq_offsets[i]);
                Log.d(TAG,"mapping index "+freq_offsets[i]);
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

        String msg = "values: ";

        for (int i = 0; i <= 2 * searchwindow; i++) {
            template[i] = 0.9 * template[i] + 0.1 * (data[basefreq-searchwindow+i] / maxval);
            msg += (int)(100 * data[basefreq - searchwindow + i] / maxval)+",";
            data[basefreq - searchwindow + i] -= maxval * template[i];
        }
        Log.d(TAG,msg);
        msg = "template: ";
        for (double t: template) msg+=(int)(t*100)+",";
        Log.d(TAG,msg);

        // find weight distribution of peak
        double balance = 0.0;
        for (int i = 5; i <= searchwindow; i++)
            balance += (data[basefreq+i] - data[basefreq-i]);

        Log.v(TAG,"basefreq = "+basefreq+" balance = "+balance);
        return (int)Math.round(balance);
    }

    // Requesting permission to RECORD_AUDIO (from https://developer.android.com/guide/topics/media/mediarecorder#java)
    private boolean permissionToRecordAccepted = false;
    private final String[] permissions = {Manifest.permission.RECORD_AUDIO,
            Manifest.permission.WRITE_EXTERNAL_STORAGE};
    private static final int REQUEST = 200;

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST) {
            permissionToRecordAccepted = (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED);
        }
        if (!permissionToRecordAccepted ) {
            Toast.makeText(this, "The app has no permissions", Toast.LENGTH_LONG).show();
            finish();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        requestPermissions(permissions, REQUEST);

        // TODO: setAudioAttributes()?
        soundPool = new SoundPool.Builder().build();
        pings = new int[6];
        pings[0] = soundPool.load(this,R.raw.sine19500,0);
        pings[1] = soundPool.load(this,R.raw.sine19700,0);
        pings[2] = soundPool.load(this,R.raw.sine19900,0);
        pings[3] = soundPool.load(this,R.raw.sine20100,0);
        pings[4] = soundPool.load(this,R.raw.sine20300,0);
        pings[5] = soundPool.load(this,R.raw.sine20500,0);

        // TODO these sounds don't have ramp-up/ramp-down yet

        fft = new KISSFastFourierTransformer();

        int bufferSize = AudioRecord.getMinBufferSize( SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        Log.d(TAG,"minBufferSize = "+bufferSize);
        audioRecord = new AudioRecord(MediaRecorder.AudioSource.UNPROCESSED, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufferSize*4 );

        rawbuffer = new short[fftwindowsize];
        input = new double[fftwindowsize];
        prev = new double[fftwindowsize];
        hann = hann_window(fftwindowsize);
        scratch = new double[fftwindowsize];

        masterbuf = new double[SAMPLE_RATE]; // room for one second of data

        FloatingActionButton fab = (FloatingActionButton) findViewById(R.id.fab);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (doRecord) {
                    doRecord = false;
                    ((FloatingActionButton)view).setImageResource(android.R.drawable.ic_media_play);
                    showYesNoDialog();
                }
                else {
                    ra = (RecordAudioTask) new RecordAudioTask().execute();
                    ((FloatingActionButton)view).setImageResource(android.R.drawable.ic_media_pause);
                }
            }
        });
/*
        imageView = (ImageView) this.findViewById(R.id.imageView);
        bitmap = Bitmap.createBitmap(canvas_size, canvas_size, Bitmap.Config.ARGB_8888);
        canvas = new Canvas(bitmap);
        paint = new Paint();
        paint.setColor(Color.GREEN);
        imageView.setImageBitmap(bitmap);
*/
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

        @Override protected Void doInBackground(Void... params) {
            try {

                audioRecord.startRecording();
                Log.d(TAG,"start recording");
                doRecord = true;

                // added
                byte[] raw_buffer = new byte[fftwindowsize];
                String filename = getTempFileName();
                FileOutputStream os = null;

                try {
                    os = new FileOutputStream(filename);
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                }
                //added


                int counter = 0;
                while (doRecord) {

                    if (counter == numberOfIterations) {
                        runOnUiThread(new Runnable()
                        {
                            public void run()
                            {
                                FloatingActionButton fab = (FloatingActionButton) findViewById(R.id.fab);
                                fab.performClick();
                            }
                        });
                    }
                    counter++;

                    // FIXME: ugly hack, when exactly should playback happen?
                    if (count++ % 3 == 0) {
                        nextfreq = (int)(Math.random()*pings.length);
                        soundPool.play(pings[nextfreq],1.0f,1.0f,0,0,1.0f );
                    }

                    // added ->
                    if (os != null) {
                        int read = audioRecord.read(raw_buffer, 0, raw_buffer.length, AudioRecord.READ_BLOCKING); // ???

                        if (read != AudioRecord.ERROR_INVALID_OPERATION) {
                            try {
                                os.write(raw_buffer);
                            } catch (IOException e) {
                                e.printStackTrace();
                            }
                        }
                    }


                    //added <-


                    long time1 = System.currentTimeMillis();
                    double[] tmpb = prev; prev = input; input = tmpb;
                    // FIXME: magic scale factor 100.0
                    for (int i = 0; i < input.length; i++) input[i] = 100.0 * (raw_buffer[i] / (double)Short.MAX_VALUE);
                    double[] output = fft_with_hann(input,0);
                    long time2 = System.currentTimeMillis();

                    Log.v(TAG,"timediff = ms: "+(time2-time1));


/*

                    // replace this part!!

                    int balance = detect_freq(output);
                    if (balance == 0) continue;

                    publishProgress(output);

                    double delta = balance - wma[nextfreq];
                    double alpha = 0.1;
                    wma[nextfreq] = wma[nextfreq] + alpha * delta;
                    wmv[nextfreq] = (1.0-alpha) * (wmv[nextfreq] + alpha*delta*delta);

                    String msg = "balance moving average: ";
                    for (double w: wma) msg+=(int)w+" ";
                    Log.d(TAG,msg);

                    msg = "balance moving stddev: ";
                    for (double v: wmv) msg+=(int)Math.sqrt(v)+" ";
                    Log.d(TAG,msg);

                    if (delta > 1.5*Math.sqrt(wmv[nextfreq])) {
                        eventcount += 1;
                    } else eventcount = 0;

                    if (eventcount >= 2) {
                        Log.d(TAG, "incoming!");
                        clearColor = Color.RED;
                    }
*/
                    // do tu
                }

                try {
                    assert os != null;
                    os.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }

                audioRecord.stop();

                // TODO: DODANO

                Log.d(TAG,"stop recording");

            } catch(Exception e) {
                e.printStackTrace();
            }
            return null;
        }

        /*
        // https://stackoverflow.com/questions/5511250/capturing-sound-for-analysis-and-visualizing-frequencies-in-android
        @Override protected void onProgressUpdate(double[]... data) {
            canvas.drawColor(clearColor);
            clearColor = Color.BLACK;
            for (int x = 0; x < canvas_size; x++) {
                // visualize only the uppermost part of the spectrum
                int startbin = (data[0].length/2) - canvas_size;
                int y1 = (int) (canvas_size - (data[0][startbin+x] * 10));
                int y2 = canvas_size;
                canvas.drawLine(x, y1, x, y2, paint);
            }
            imageView.invalidate();
        }
        */
    }

    private void showYesNoDialog() {

        DialogInterface.OnClickListener dialogClickListener = new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                switch (which) {
                    case DialogInterface.BUTTON_POSITIVE:
                        isPositive = "_1";
                        break;

                    case DialogInterface.BUTTON_NEGATIVE:
                        isPositive = "_0";
                        break;
                }
                copyWaveFile(getTempFileName(), getFileName());
                deleteTempFile();
            }
        };

        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setMessage("Did you bump into a wall?!").setPositiveButton("Yes", dialogClickListener)
                .setNegativeButton("No", dialogClickListener).show();


    }

    private String getTempFileName() {
        String filepath = Environment.getExternalStorageDirectory().getPath();
        File file = new File(filepath, AUDIO_RECORDER_FOLDER);

        if (!file.exists())
            file.mkdirs();

        File tempfile = new File(filepath, AUDIO_RECORDER_TEMP_FILE);

        if (tempfile.exists())
            tempfile.delete();

        return (file.getAbsolutePath() + "/" + AUDIO_RECORDER_TEMP_FILE);
    }

    private String getFileName() {
        String filepath = Environment.getExternalStorageDirectory().getPath();
        File file = new File(filepath, AUDIO_RECORDER_FOLDER);

        if(!file.exists())
            file.mkdirs();

        return(file.getAbsolutePath() + "/" + System.currentTimeMillis() + isPositive + AUDIO_RECORDER_EXT_FILE);
    }

    private void deleteTempFile() {
        File file = new File(getTempFileName());
        file.delete();
    }

    private void copyWaveFile(String inFilename, String outFilename) {
        FileInputStream in;
        FileOutputStream out;

        int bufferSize = AudioRecord.getMinBufferSize( SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT );

        long totalAudioLen, totalDataLen;
        long longSampleRate = SAMPLE_RATE;
        int channels = 2;
        long byteRate = RECORDER_BPP * SAMPLE_RATE * channels / 8;

        byte[] data = new byte[bufferSize];

        try {
            in = new FileInputStream(inFilename);
            out = new FileOutputStream(outFilename);
            totalAudioLen = in.getChannel().size();
            totalDataLen = totalAudioLen + 36;

            Log.v(TAG, "File size: " + totalDataLen);

            WriteWaveFileHeader(out, totalAudioLen, totalDataLen, longSampleRate, channels, byteRate);

            while(in.read(data) != -1) {
                out.write(data);
            }

            in.close();
            out.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void WriteWaveFileHeader(FileOutputStream out, long totalAudioLen, long totalDataLen, long longSampleRate, int channels, long byteRate) throws IOException {
        byte[] header = new byte[44];

        header[0] = 'R';  // RIFF/WAVE header
        header[1] = 'I';
        header[2] = 'F';
        header[3] = 'F';
        header[4] = (byte) (totalDataLen & 0xff);
        header[5] = (byte) ((totalDataLen >> 8) & 0xff);
        header[6] = (byte) ((totalDataLen >> 16) & 0xff);
        header[7] = (byte) ((totalDataLen >> 24) & 0xff);
        header[8] = 'W';
        header[9] = 'A';
        header[10] = 'V';
        header[11] = 'E';
        header[12] = 'f';  // 'fmt ' chunk
        header[13] = 'm';
        header[14] = 't';
        header[15] = ' ';
        header[16] = 16;  // 4 bytes: size of 'fmt ' chunk
        header[17] = 0;
        header[18] = 0;
        header[19] = 0;
        header[20] = 1;  // format = 1
        header[21] = 0;
        header[22] = (byte) channels;
        header[23] = 0;
        header[24] = (byte) (longSampleRate & 0xff);
        header[25] = (byte) ((longSampleRate >> 8) & 0xff);
        header[26] = (byte) ((longSampleRate >> 16) & 0xff);
        header[27] = (byte) ((longSampleRate >> 24) & 0xff);
        header[28] = (byte) (byteRate & 0xff);
        header[29] = (byte) ((byteRate >> 8) & 0xff);
        header[30] = (byte) ((byteRate >> 16) & 0xff);
        header[31] = (byte) ((byteRate >> 24) & 0xff);
        header[32] = (byte) (2 * 16 / 8);  // block align
        header[33] = 0;
        header[34] = RECORDER_BPP;  // bits per sample
        header[35] = 0;
        header[36] = 'd';
        header[37] = 'a';
        header[38] = 't';
        header[39] = 'a';
        header[40] = (byte) (totalAudioLen & 0xff);
        header[41] = (byte) ((totalAudioLen >> 8) & 0xff);
        header[42] = (byte) ((totalAudioLen >> 16) & 0xff);
        header[43] = (byte) ((totalAudioLen >> 24) & 0xff);

        out.write(header, 0, 44);
    }
}