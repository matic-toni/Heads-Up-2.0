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
import android.os.Environment;
import android.support.annotation.NonNull;
import android.support.design.widget.FloatingActionButton;
import android.support.v4.app.NotificationCompat;
import android.support.v4.app.NotificationManagerCompat;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.View;
import android.view.Menu;
import android.widget.TextView;
import android.widget.Toast;

import uk.me.berndporr.kiss_fft.KISSFastFourierTransformer;
import org.apache.commons.math3.complex.Complex;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {

    public static String TAG = "DistriBat";

    private static final String AUDIO_RECORDER_FOLDER = "AudioRecorder";
    private static final String AUDIO_RECORDER_EXT_FILE = ".wav";
    private static final String AUDIO_RECORDER_TEMP_FILE = "record_temp";
    private static final String AUDIO_RECORDER_TEMP_FILE_EXT = ".raw";
    private static final int RECORDER_BPP = 16;

    // number of buffers recorded into one wav file. If you want longer audios, increase this number
    private static final int NUMBER_OF_BUFFERS = 1;

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

    short[] rawbuffer;
    double[] input;
    double[] prev;
    double[] hann;
    double[] scratch;
    double[] masterbuf;

    boolean stopped = false;
    boolean doRecord = false;

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

    // detect "interesting" frequencies in FFT result
    // TODO: Do we need this?!?
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

        StringBuilder msg = new StringBuilder("values: ");

        for (int i = 0; i <= 2 * searchwindow; i++) {
            template[i] = 0.9 * template[i] + 0.1 * (data[basefreq-searchwindow+i] / maxval);
            msg.append((int) (100 * data[basefreq - searchwindow + i] / maxval)).append(",");
            data[basefreq - searchwindow + i] -= maxval * template[i];
        }
        Log.d(TAG, msg.toString());
        msg = new StringBuilder("template: ");
        for (double t: template) msg.append((int) (t * 100)).append(",");
        Log.d(TAG, msg.toString());

        // find weight distribution of peak
        double balance = 0.0;
        for (int i = 5; i <= searchwindow; i++)
            balance += (data[basefreq+i] - data[basefreq-i]);

        Log.v(TAG,"basefreq = "+basefreq+" balance = "+balance);
        return (int)Math.round(balance);
    }

    // Requesting permission to RECORD_AUDIO (from https://developer.android.com/guide/topics/media/mediarecorder#java)
    // Added permission for saving files into a local memory
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

        Toolbar toolbar = findViewById(R.id.toolbar);
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
        Log.d(TAG,"minBufferSize = " + bufferSize);
        audioRecord = new AudioRecord(MediaRecorder.AudioSource.UNPROCESSED, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufferSize*4 );

        rawbuffer = new short[fftwindowsize];
        input = new double[fftwindowsize];
        prev = new double[fftwindowsize];
        hann = hann_window(fftwindowsize);
        scratch = new double[fftwindowsize];

        masterbuf = new double[SAMPLE_RATE]; // room for one second of data

        FloatingActionButton fab = findViewById(R.id.fab);
        final TextView textView = findViewById(R.id.statusText);

        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (doRecord) {
                    stopped = true;
                    doRecord = false;
                    ra = (RecordAudioTask) new RecordAudioTask().execute();
                    ((FloatingActionButton)view).setImageResource(android.R.drawable.ic_media_play);
                    String status = "Recording sopped!\n\nYou will no longer receive notifications!";
                    textView.setText(status);
                }
                else {
                    stopped = false;
                    doRecord = true;
                    ra = (RecordAudioTask) new RecordAudioTask().execute();
                    ((FloatingActionButton)view).setImageResource(android.R.drawable.ic_media_pause);
                    String status = "Recording started...\n\nYou can close the app now!";
                    // TODO: Ja bi ode možda crta te spektograme ako su baza, da nije prazan ekran.
                    textView.setText(status);
                }
            }
        });
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    // I commented out a settings option from the top-right corner. It was pointless.
    /*
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
     */

    // https://www.androidcookbook.info/android-media/visualizing-frequencies.html
    private class RecordAudioTask extends AsyncTask<Void, double[], Void> {

        @Override protected Void doInBackground(Void... params) {
            while (!stopped) {
                try {
                    audioRecord.startRecording();
                    Log.d(TAG,"start recording");
                    doRecord = true;

                    byte[] raw_buffer = new byte[fftwindowsize];

                    String filename = getTempFileName("_0");
                    String filename_han = getTempFileName("_1");

                    FileOutputStream os = new FileOutputStream(filename);
                    FileOutputStream os_han = new FileOutputStream(filename_han);

                    int counter = 0;
                    while (doRecord) {

                        // Checks if there is enough data in output stream. If yes - stop recording
                        if (counter == NUMBER_OF_BUFFERS) {
                            // todo: what happens then?
                            break;
                        }
                        counter++;

                        // FIXME: ugly hack, when exactly should playback happen?
                        if (count++ % 3 == 0) {
                            nextfreq = (int)(Math.random()*pings.length);
                            soundPool.play(pings[nextfreq],1.0f,1.0f,0,0,1.0f );
                        }

                        // Copy data from buffer to output stream (== temp file)
                        int read = audioRecord.read(raw_buffer, 0, raw_buffer.length, AudioRecord.READ_BLOCKING);

                        if (read != AudioRecord.ERROR_INVALID_OPERATION) {
                            os.write(raw_buffer);
                        }

                        long time1 = System.currentTimeMillis();
                        double[] tmpb = prev;
                        prev = input;
                        input = tmpb;

                        // FIXME: magic scale factor 100.0
                        for (int i = 0; i < input.length; i++)
                            input[i] = 100.0 * (raw_buffer[i] / (double)Short.MAX_VALUE);

                        double[] output = fft_with_hann(input,0);

                        // added - converting doubles to bytes
                        // ! this wav files will be 8 times larger, since it is using doubles
                        // instead of bytes
                        ByteBuffer bb = ByteBuffer.allocate(output.length * 8);
                        for(double o : output) {
                            bb.putDouble(o);
                        }

                        byte[] output_bytes = bb.array();

                        os_han.write(output_bytes);

                        long time2 = System.currentTimeMillis();

                        Log.v(TAG,"timediff = ms: "+(time2-time1));
                    }

                    saveFiles();

                    // TODO: Classification goes here!
                    runOnUiThread(new Runnable()
                    {
                        public void run()
                        {
                            final TextView textView = findViewById(R.id.statusText);

                            double rand = Math.random();
                            if (rand > 0.95) {

                                NotificationCompat.Builder builder = new NotificationCompat.Builder(MainActivity.this,
                                        "My Notification");

                                builder.setContentTitle("STOP!!!")
                                        .setContentText("The wall is in front of you!")
                                        .setSmallIcon(R.drawable.app_logo)
                                        .setAutoCancel(true);

                                NotificationManagerCompat manager = NotificationManagerCompat.from(MainActivity.this);
                                manager.notify(1, builder.build());

                                // This is important for Android versions Oreo and older.
                                // Not sure if it has to be in the onCreate() function.
                                /*
                                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                                    NotificationChannel channel = new NotificationChannel("My Notification", "My Notificiation", NotificationManager.IMPORTANCE_HIGH);
                                    NotificationManager man = getSystemService(NotificationManager.class);
                                    assert man != null;
                                    man.createNotificationChannel(channel);
                                }

                                String status = "Watch Your Head!";
                                textView.setText(status);
                                textView.setTextColor(Color.RED);
                            } else {
                                String status = "Walk...";
                                textView.setText(status);
                                textView.setTextColor(Color.GREEN);

                                */

                            }

                        }
                    });
                    // TODO: End of Classification



                    // TODO: deleteFiles();

                    // Closing the output stream
                    os.close();
                    os_han.close();

                    audioRecord.stop();

                    Log.d(TAG,"stop recording");

                } catch(Exception e) {
                    e.printStackTrace();
                }

            }

            deleteFiles();
            return null;
        }
    }

    private void saveFiles() {
        copyWaveFile(getTempFileName("_0"), getFileName("_0"));
        copyWaveFile(getTempFileName("_1"), getFileName("_1"));

    }


    // Makes #NUMBER_OF_SAMPLES temp files which will be copied to wav files later.
    private String getTempFileName(String isHan) {
        String filepath = Environment.getExternalStorageDirectory().getPath();
        File file = new File(filepath, AUDIO_RECORDER_FOLDER);

        if (!file.exists())
            if (!file.mkdirs())
                Log.v(TAG, "An Error has Occurred!");

        File temp_file = new File(filepath, AUDIO_RECORDER_TEMP_FILE + AUDIO_RECORDER_TEMP_FILE_EXT);

        if (temp_file.exists())
            if (!temp_file.delete())
                Log.v(TAG, "An Error Has Occurred!");

        return (file.getAbsolutePath() + "/" + AUDIO_RECORDER_TEMP_FILE + isHan + AUDIO_RECORDER_TEMP_FILE_EXT);
    }

    // Creates one unique file name for new recorded sample
    // Every file name ends with underscore followed by number 0 or 1
    // This number indicates if the sample is positive or negative
    private String getFileName(String isHan) {
        String filepath = Environment.getExternalStorageDirectory().getPath();
        File file = new File(filepath, AUDIO_RECORDER_FOLDER);

        if(!file.exists())
            if (!file.mkdirs())
                Log.v(TAG, "An Error Has Occurred!");

        return(file.getAbsolutePath() + "/" + System.currentTimeMillis() + isHan + AUDIO_RECORDER_EXT_FILE);
    }

    private void deleteFiles() {
        String dirPath = Environment.getExternalStorageDirectory().getPath();
        File dir = new File(dirPath, AUDIO_RECORDER_FOLDER);

        for (File child : dir.listFiles())
            if (!child.delete())
                Log.v(TAG, "An Error Has Occurred");
    }

    // Copies one temp file to a one .wav file
    private void copyWaveFile(String inFilename, String outFilename) {
        FileInputStream in;
        FileOutputStream out;

        int bufferSize = AudioRecord.getMinBufferSize( SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT );

        long totalAudioLen, totalDataLen;
        long longSampleRate = SAMPLE_RATE;
        int channels = 2;
        long byteRate = (long) RECORDER_BPP * SAMPLE_RATE * channels / 8;

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

    // Writes wave header at the start of the new wave file
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