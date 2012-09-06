#r "Microsoft.VisualBasic"
#load "Data.fs"
#load "Validation.fs"
#load "NaiveBayes.fs"
System.IO.Directory.SetCurrentDirectory(__SOURCE_DIRECTORY__)

open Charon.Data
open Charon.Validation
open MachineLearning.NaiveBayes
open Microsoft.VisualBasic.FileIO

#time

let trainSampleSet = @"..\..\..\train-sample.csv"
let publicLeaderboard = @"..\..\..\public_leaderboard.csv"

// split the data into train and test sets as 75/25
let trainPct = 0.75

// indices of the title and body columns
let titleCol, bodyCol = 6, 7

// retrieve OpenStatus and given column
let getQuestionsData setFileName col =
    parseCsv setFileName
    |> Seq.skip 1
    |> Seq.map (fun line -> line.[14], line.[col])
    |> Seq.toList

let getPublicData publicLeaderboard= 
    parseCsv publicLeaderboard
    |> Seq.skip 1
    |> Seq.map (fun line -> line.[7])
    |> Seq.toList

let inline size pct len = int(ceil(pct * float len))

// split data into train and test sets
let splitSets fileName trainPct col =
    let questionsData = getQuestionsData fileName col
    let sampleSize = size trainPct questionsData.Length
    questionsData
    |> Seq.fold (fun (i, (sample, test)) q -> 
        if i <= sampleSize then i+1, (q::sample, test)
        else i+1, (sample, q::test)) (1,([],[]))
    |> snd

// dumb classifier, matches every category to prior proba of each class
let priorClassifier category =
    match category with
    | "not a real question" -> priors
    | "not constructive" -> priors
    | "off topic" -> priors
    | "open" -> priors
    | "too localized" -> priors
    | _ -> failwith "Unrecognized category"

// need to do some work on this to produce probabilities
// based on Bayes classifier output. This is raw, based on
// observed error using classifier
let bodyClassifier category =
    match category with
    | "not a real question" ->
        [ ("not a real question", 0.397); ("not constructive", 0.146); ("off topic", 0.096); ("open", 0.276); ("too localized", 0.086) ]
    | "not constructive" ->
        [ ("not a real question", 0.124); ("not constructive", 0.082); ("off topic", 0.105); ("open", 0.582); ("too localized", 0.108) ]
    | "off topic" ->
        [ ("not a real question", 0.103); ("not constructive", 0.237); ("off topic", 0.483); ("open", 0.158); ("too localized", 0.02) ]
    | "open" ->
        [ ("not a real question", 0.0091347705760047); ("not constructive", 0.0046458596397953); ("off topic", 0.00520096554605094); ("open", 0.979191390785063); ("too localized", 0.00182701345308509) ]
    | "too localized" ->
        [ ("not a real question", 0.162); ("not constructive", 0.042); ("off topic", 0.098); ("open", 0.363); ("too localized", 0.336) ]
    | _ -> failwith "Unrecognized category"

//let trainSample trainSet =
//    let tokens = topByClass trainSet 500
//    let training = train setOfWords trainSet tokens
//    classify training
//    //todo: return probabilities here
//
//
//// Visualize classification results by group
//let visualizeByGroup test testSet =
//    testSet
//    |> Seq.map (fun (c, t) -> 
//        let result = test t |> Seq.maxBy snd |> fst
//        c, result)
//    |> Seq.groupBy fst
//    |> Seq.map (fun (cl, gr) ->
//        let grouped = gr |> Seq.groupBy snd
//        cl,
//        grouped |> Seq.map (fun (res, cases) -> res, Seq.length cases))
//    |> Seq.iter (fun (cl, results) -> 
//        printfn ""
//        printfn "Real: %s" cl
//        results |> Seq.iter (fun (g, c) -> printfn "%s, %i" g c))

// Evaluate % correctly classified

//questionTitles
//|> Seq.skip (sampleSize + 1)
//|> Seq.take 1000
//|> Seq.map (fun (c, t) -> 
//    let result = 
//        test t |> Seq.maxBy snd |> fst
//    c, if result = c then 1.0 else 0.0)
//|> Seq.groupBy fst
//|> Seq.map (fun (cl, gr) -> cl, gr |> Seq.averageBy snd)
//|> Seq.iter (fun (cl, prob) -> printfn "%s %f" cl prob)

// http://www.textfixer.com/resources/common-english-words.txt
let stopWords = "a,able,about,across,after,all,almost,also,am,among,an,and,any,are,as,at,be,because,been,but,by,can,cannot,could,dear,did,do,does,either,else,ever,every,for,from,get,got,had,has,have,he,her,hers,him,his,how,however,i,if,in,into,is,it,its,just,least,let,like,likely,may,me,might,most,must,my,neither,no,nor,not,of,off,often,on,only,or,other,our,own,rather,said,say,says,she,should,since,so,some,than,that,the,their,them,then,there,these,they,this,tis,to,too,twas,us,wants,was,we,were,what,when,where,which,while,who,whom,why,will,with,would,yet,you,your"
let remove = stopWords.Split(',') |> Set.ofArray

//let words = 
//        dataset
//        |> extractWords
//        |> Set.filter (fun w -> remove.Contains(w) |> not)

printfn "Reading data sets"
let trainSet, testSet = splitSets trainSampleSet trainPct bodyCol

printfn "Training model on training set"

let test =
    let tokens = topByClass trainSet 500
    train setOfWords trainSet tokens
let classifier = classify test
let model = fun (text:string) -> (classifier text |> renormalize);;

printfn "Saving"
saveWordsFrequencies @"..\..\..\bayes.csv" test

//printfn "Testing model on test set"
let inline predict model = Seq.map model 
let predictions = 
    predict model (Seq.map snd testSet) 
    |> Seq.map (fun e -> Map.toSeq e)

//printfn "Analyze"
//visualizeByGroup model testSet
quality predictions (Seq.map fst testSet)

// printfn "Reading"
// let back = readWordsFrequencies @"..\..\..\bayes.csv";;
// let model = updatePriors back priors

/// following block is a question which fails to classify
// Question 46 of sample set classifies as NaN??? See below

// "I have created one application which record the sound and also play that sound.
//While i m using it from Windows xp is works fine. But while i am using Windows 7 for the Same code then it gives me error.
//The Code that i use to record the Sound id:
//    record.setOnClickListener(new View.OnClickListener() 
//        {
//        	boolean mStartRecording = true;
//        	public void onClick(View v) 
//			{
//				//onRecord(mStartRecording);
//        		if(wordValue.getText().toString().equals(""))
//        		{
//        			Toast.makeText(getApplicationContext(), "Please Insert word Value", Toast.LENGTH_SHORT).show();
//        		}
//        		else
//        		{
//        			if (mStartRecording==true) 
//        			{
//        				//startRecording();
//        				haveStartRecord=true;
//        				String recordWord = wordValue.getText().toString();
//        				String file = Environment.getExternalStorageDirectory().getAbsolutePath();
//        				file = file+"/"+recordWord+".3gp";
//        				System.out.println("Recording Start");
//        				//record.setText("Stop recording");
//        				record.setBackgroundDrawable(getResources().getDrawable( R.drawable.rec_on));
//        				mRecorder = new MediaRecorder();
//        				mRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
//        				mRecorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP); 
//        				mRecorder.setOutputFile(file);
//        				mRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
//        				try 
//        				{
//        					mRecorder.prepare();
//        				}
//        				catch (IOException e) 
//        				{
//        					Log.e(LOG_TAG, "prepare() failed");
//        				}
//        				mRecorder.start();
//        			}
//        			else
//        			{
//        				//stopRecording();
//        				System.out.println("Recording Stop");
//        				record.setBackgroundDrawable(getResources().getDrawable( R.drawable.rec_off));
//        				mRecorder.stop();
//        				mRecorder.release();
//        				mRecorder = null;
//        				haveFinishRecord=true;
//        			}
//        			mStartRecording = !mStartRecording;
//        		}
//			}
//		});
//And the Error log i got while i am record that sound from Windows7 is:
//    09-07 15:12:40.365: ERROR/audio_input(34): unsupported parameter: x-pvmf/media-input-node/cap-config-interface;valtype=key_specific_value
//09-07 15:12:40.365: ERROR/audio_input(34): VerifyAndSetParameter failed
//09-07 15:12:40.415: ERROR/PVOMXEncNode(34): PVMFOMXEncNode-Audio_AMRNB::DoPrepare(): Got Component OMX.PV.amrencnb handle 
//09-07 15:12:40.525: ERROR/AudioFlinger(34): Error reading audio input
//09-07 15:12:47.315: WARN/AudioRecord(34): obtainBuffer timed out (is the CPU pegged?) user=00000000, server=00000000
//09-07 15:12:47.325: ERROR/AudioFlinger(34): Error reading audio input
//09-07 15:12:52.756: WARN/WindowManager(59): Key dispatching timed out sending to com.quiz.spellingquiz/com.quiz.spellingquiz.EnterWordsActivity
//09-07 15:12:52.756: WARN/WindowManager(59): Previous dispatch state: {{KeyEvent{action=1 code=66 repeat=0 meta=0 scancode=28 mFlags=8} to Window{450544a8 com.quiz.spellingquiz/com.quiz.spellingquiz.EnterWordsActivity paused=false} @ 1315388558729 lw=Window{450544a8 com.quiz.spellingquiz/com.quiz.spellingquiz.EnterWordsActivity paused=false} lb=android.os.BinderProxy@44f25118 fin=false gfw=true ed=true tts=0 wf=false fp=false mcf=Window{450544a8 com.quiz.spellingquiz/com.quiz.spellingquiz.EnterWordsActivity paused=false}}}
//09-07 15:12:52.807: WARN/WindowManager(59): Current dispatch state: {{null to Window{450544a8 com.quiz.spellingquiz/com.quiz.spellingquiz.EnterWordsActivity paused=false} @ 1315388572808 lw=Window{450544a8 com.quiz.spellingquiz/com.quiz.spellingquiz.EnterWordsActivity paused=false} lb=android.os.BinderProxy@44f25118 fin=false gfw=true ed=true tts=0 wf=false fp=false mcf=Window{450544a8 com.quiz.spellingquiz/com.quiz.spellingquiz.EnterWordsActivity paused=false}}}
//09-07 15:12:52.836: INFO/Process(59): Sending signal. PID: 1979 SIG: 3
//09-07 15:12:52.836: INFO/dalvikvm(1979): threadid=3: reacting to signal 3
//09-07 15:12:52.855: INFO/dalvikvm(1979): Wrote stack traces to '/data/anr/traces.txt'
//09-07 15:12:52.855: INFO/Process(59): Sending signal. PID: 59 SIG: 3
//09-07 15:12:52.855: INFO/dalvikvm(59): threadid=3: reacting to signal 3
//09-07 15:12:52.942: INFO/dalvikvm(59): Wrote stack traces to '/data/anr/traces.txt'
//09-07 15:12:52.946: INFO/Process(59): Sending signal. PID: 120 SIG: 3
//09-07 15:12:52.946: INFO/dalvikvm(120): threadid=3: reacting to signal 3
//09-07 15:12:52.967: INFO/dalvikvm(120): Wrote stack traces to '/data/anr/traces.txt'
//09-07 15:12:52.978: INFO/Process(59): Sending signal. PID: 128 SIG: 3
//09-07 15:12:52.986: INFO/dalvikvm(128): threadid=3: reacting to signal 3
//09-07 15:12:52.995: INFO/dalvikvm(128): Wrote stack traces to '/data/anr/traces.txt'
//09-07 15:12:53.005: INFO/Process(59): Sending signal. PID: 325 SIG: 3
//09-07 15:12:53.005: INFO/dalvikvm(325): threadid=3: reacting to signal 3
//09-07 15:12:53.035: INFO/dalvikvm(325): Wrote stack traces to '/data/anr/traces.txt'
//09-07 15:12:53.045: INFO/Process(59): Sending signal. PID: 316 SIG: 3
//09-07 15:12:53.045: INFO/dalvikvm(316): threadid=3: reacting to signal 3
//09-07 15:12:53.065: INFO/dalvikvm(316): Wrote stack traces to '/data/anr/traces.txt'
//09-07 15:12:53.077: INFO/Process(59): Sending signal. PID: 192 SIG: 3
//09-07 15:12:53.077: INFO/dalvikvm(192): threadid=3: reacting to signal 3
//09-07 15:12:53.096: INFO/dalvikvm(192): Wrote stack traces to '/data/anr/traces.txt'
//09-07 15:12:53.105: INFO/Process(59): Sending signal. PID: 125 SIG: 3
//09-07 15:12:53.105: INFO/dalvikvm(125): threadid=3: reacting to signal 3
//09-07 15:12:53.155: INFO/dalvikvm(125): Wrote stack traces to '/data/anr/traces.txt'
//09-07 15:12:53.165: INFO/Process(59): Sending signal. PID: 243 SIG: 3
//09-07 15:12:53.165: INFO/dalvikvm(243): threadid=3: reacting to signal 3
//09-07 15:12:53.186: WARN/AudioRecord(34): obtainBuffer timed out (is the CPU pegged?) user=00000000, server=00000000
//09-07 15:12:53.186: ERROR/AudioFlinger(34): Error reading audio input
//09-07 15:12:53.186: WARN/audio_input(34): numOfBytes (0) <= 0.
//09-07 15:12:53.195: INFO/dalvikvm(243): Wrote stack traces to '/data/anr/traces.txt'
//09-07 15:12:53.207: INFO/Process(59): Sending signal. PID: 234 SIG: 3
//09-07 15:12:53.207: INFO/dalvikvm(234): threadid=3: reacting to signal 3
//09-07 15:12:53.215: INFO/dalvikvm(234): Wrote stack traces to '/data/anr/traces.txt'
//09-07 15:12:53.226: INFO/Process(59): Sending signal. PID: 252 SIG: 3
//09-07 15:12:53.226: INFO/dalvikvm(252): threadid=3: reacting to signal 3
//09-07 15:12:53.245: INFO/dalvikvm(252): Wrote stack traces to '/data/anr/traces.txt'
//09-07 15:12:53.255: INFO/Process(59): Sending signal. PID: 225 SIG: 3
//09-07 15:12:53.255: INFO/dalvikvm(225): threadid=3: reacting to signal 3
//09-07 15:12:53.275: INFO/dalvikvm(225): Wrote stack traces to '/data/anr/traces.txt'
//09-07 15:12:53.286: INFO/Process(59): Sending signal. PID: 202 SIG: 3
//09-07 15:12:53.286: INFO/dalvikvm(202): threadid=3: reacting to signal 3
//09-07 15:12:53.295: INFO/dalvikvm(202): Wrote stack traces to '/data/anr/traces.txt'
//09-07 15:12:53.306: INFO/Process(59): Sending signal. PID: 172 SIG: 3
//09-07 15:12:53.306: INFO/dalvikvm(172): threadid=3: reacting to signal 3
//09-07 15:12:53.317: INFO/dalvikvm(172): Wrote stack traces to '/data/anr/traces.txt'
//09-07 15:12:53.325: INFO/Process(59): Sending signal. PID: 164 SIG: 3
//09-07 15:12:53.338: INFO/dalvikvm(164): threadid=3: reacting to signal 3
//09-07 15:12:53.356: INFO/dalvikvm(164): Wrote stack traces to '/data/anr/traces.txt'
//09-07 15:12:53.356: INFO/Process(59): Sending signal. PID: 131 SIG: 3
//09-07 15:12:53.365: INFO/dalvikvm(131): threadid=3: reacting to signal 3
//09-07 15:12:53.386: INFO/dalvikvm(131): Wrote stack traces to '/data/anr/traces.txt'
//09-07 15:12:53.455: ERROR/ActivityManager(59): ANR in com.quiz.spellingquiz (com.quiz.spellingquiz/.EnterWordsActivity)
//09-07 15:12:53.455: ERROR/ActivityManager(59): Reason: keyDispatchingTimedOut
//09-07 15:12:53.455: ERROR/ActivityManager(59): Load: 0.75 / 0.6 / 0.49
//09-07 15:12:53.455: ERROR/ActivityManager(59): CPU usage from 231476ms to 53ms ago:
//09-07 15:12:53.455: ERROR/ActivityManager(59):   system_server: 16% = 13% user + 3% kernel / faults: 9413 minor
//09-07 15:12:53.455: ERROR/ActivityManager(59):   ronsoft.openwnn: 2% = 2% user + 0% kernel / faults: 1992 minor
//09-07 15:12:53.455: ERROR/ActivityManager(59):   adbd: 2% = 0% user + 2% kernel / faults: 651 minor
//09-07 15:12:53.455: ERROR/ActivityManager(59):   id.defcontainer: 1% = 0% user + 0% kernel / faults: 167 minor
//09-07 15:12:53.455: ERROR/ActivityManager(59):   ndroid.launcher: 0% = 0% user + 0% kernel / faults: 1711 minor
//09-07 15:12:53.455: ERROR/ActivityManager(59):   d.process.acore: 0% = 0% user + 0% kernel / faults: 2389 minor
//09-07 15:12:53.455: ERROR/ActivityManager(59):   m.android.phone: 0% = 0% user + 0% kernel / faults: 22 minor
//09-07 15:12:53.455: ERROR/ActivityManager(59):   qemud: 0% = 0% user + 0% kernel
//09-07 15:12:53.455: ERROR/ActivityManager(59):   mediaserver: 0% = 0% user + 0% kernel / faults: 30 minor
//09-07 15:12:53.455: ERROR/ActivityManager(59):   com.svox.pico: 0% = 0% user + 0% kernel / faults: 49 minor
//09-07 15:12:53.455: ERROR/ActivityManager(59):   logcat: 0% = 0% user + 0% kernel
//09-07 15:12:53.455: ERROR/ActivityManager(59):   events/0: 0% = 0% user + 0% kernel
//09-07 15:12:53.455: ERROR/ActivityManager(59):   installd: 0% = 0% user + 0% kernel / faults: 6 minor
//09-07 15:12:53.455: ERROR/ActivityManager(59):   zygote: 0% = 0% user + 0% kernel / faults: 83 minor
//09-07 15:12:53.455: ERROR/ActivityManager(59):   re-initialized>: 0% = 0% user + 0% kernel / faults: 9 minor
//09-07 15:12:53.455: ERROR/ActivityManager(59):   .quicksearchbox: 0% = 0% user + 0% kernel / faults: 8 minor
//09-07 15:12:53.455: ERROR/ActivityManager(59):   m.android.music: 0% = 0% user + 0% kernel / faults: 7 minor
//09-07 15:12:53.455: ERROR/ActivityManager(59):   rild: 0% = 0% user + 0% kernel
//09-07 15:12:53.455: ERROR/ActivityManager(59):   re-initialized>: 0% = 0% user + 0% kernel / faults: 7 minor
//09-07 15:12:53.455: ERROR/ActivityManager(59):   com.android.mms: 0% = 0% user + 0% kernel / faults: 8 minor
//09-07 15:12:53.455: ERROR/ActivityManager(59):   android.protips: 0% = 0% user + 0% kernel / faults: 7 minor
//09-07 15:12:53.455: ERROR/ActivityManager(59):   ndroid.settings: 0% = 0% user + 0% kernel / faults: 7 minor
//09-07 15:12:53.455: ERROR/ActivityManager(59):   m.android.email: 0% = 0% user + 0% kernel / faults: 8 minor
//09-07 15:12:53.455: ERROR/ActivityManager(59):  +iz.spellingquiz: 0% = 0% user + 0% kernel
//09-07 15:12:53.455: ERROR/ActivityManager(59):  +iz.spellingquiz: 0% = 0% user + 0% kernel
//09-07 15:12:53.455: ERROR/ActivityManager(59): TOTAL: 38% = 27% user + 11% kernel + 0% irq + 0% softirq
//09-07 15:12:53.475: WARN/WindowManager(59): No window to dispatch pointer action 1
//09-07 15:12:53.655: DEBUG/dalvikvm(59): GC_FOR_MALLOC freed 3320 objects / 492216 bytes in 147ms
//09-07 15:12:53.915: DEBUG/dalvikvm(59): GC_FOR_MALLOC freed 312 objects / 186816 bytes in 236ms
//09-07 15:12:57.185: WARN/ActivityManager(59):   Force finishing activity com.quiz.spellingquiz/.EnterWordsActivity
//I dont know where i am wrong. And what mistake i m doing. Please let me know whats the problem while do recording for android device from windows 7 OS.
//Please help me."