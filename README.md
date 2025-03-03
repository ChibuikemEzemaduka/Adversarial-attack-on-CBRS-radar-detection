# Adversarial attack on CBRS radar detection
The Citizens Broadband Radio Service (CBRS) represents a paradigm shift in spectrum
sharing, offering enhanced bandwidth and connectivity for next-generation
wireless devices through the utilization of the underexploited 3.5 GHz frequency
band. This band, historically reserved for US government radar systems (primary
users), is now accessible to commercial wireless operators(secondry users) in the
absence of primary users. The smooth operation of CBRS depends heavily on its
precision in detecting radar system activity, where deep learning classifiers have
shown promising results. Despite considerable advancements in classification performance,
the susceptibility of these systems to adversarial learning attacks remains
under-investigated. This study addresses this gap by implementing and evaluating
the potential of adversarial attacks to compromise the CBRS radar detection system.
Such attacks could render the classifier oblivious to radar operations, precipitating
a transmission collision and consequent system breakdown. Our findings reveal the
limitations of adversarial influence, with attacks achieving efficacy predominantly
at low Signal to Noise Ratios (SNR). While classifier accuracy remains stable
above 10 dB SNR, there is a notable degradation in model confidence up to 40 %,
underscoring the need for enhanced defense strategies against adversarial attacks
in CBRS operations.

You can run the codes in the following order if you wish to start everything from scratch
OR depending on the section you are interested in, you can decide to just run the code for that section

1. TO GENERATE TRAINING DATA (SPECTROGRAM SAMPLES) FOR TRAINING THE RADAR DETECTOR

      - Run CBRS_Adversarial_attack/Train_radar_detector/Data_Generation/main.m
      
        NB: You must have MATLAB installed and the necessary toolboxes used in the run files


2. TO TRAIN THE RADAR DETECTOR

      - Run CBRS_Adversarial_attack/Train_radar_detector/Detector_training/create_train_data.py   then
      - Run CBRS_Adversarial_attack/Train_radar_detector/Detector_training/create_test_data.py    then
      - Run CBRS_Adversarial_attack/Train_radar_detector/Detector_training/main.py

        NB: You can run main.py without the first two steps if the /numpy_data folder already has files in it
            The first two steps are simply to convert the training spectrogram samples from .png to .npy format


3. TO RUN THE ADVERSARIAL ATTACK SIMULATION AND GENERATE THE ADVERSARIAL SAMPLES

     - Run CBRS_Adversarial_attack/Attack_simulation/Data_Generation/main.m 

       NB: You must have MATLAB installed and the necessary toolboxes used in the run files


4. TO EVALUATE THE EFFECTS OF THE ADVERSARIAL ATTACK ON THE RADAR DETECTOR

     - Run CBRS_Adversarial_attack/Attack_simulation/Evaluate_attack/create_test_data.py  then 
     - Run CBRS_Adversarial_attack/Attack_simulation/Evaluate_attack/main.py 

       NB: You can run main.py without the first step if the /numpy_data folder already has files in it
           The first step is simply to convert the adversarial samples from .png to .npy format
 
