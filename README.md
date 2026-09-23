# github-portfolio

# Summer's Portfolio

Welcome to my GitHub portfolio! This repository showcases my projects and coding samples, demonstrating my skills and expertise in C++ and python. It serves as a comprehensive overview of my work and accomplishments as a software engineer intern.

Resume: https://www.summer-royal.com/resume.pdf

## Silly Little Side Projects

### Memory-as-Action
- Description: A five-stage pipeline — memory bank construction, retrieval, expert annotation, SFT warm-start, then GRPO — that distills a 32B teacher into a 7B model which learns to reach for medical textbook entries while answering USMLE questions. Retrieval normally gets bolted on as a fixed step that always fires; here it is an action the model chooses, so the interesting question becomes when a small model decides it needs to look something up. Ablated against self-consistency voting, cloze scoring, and DAgger distillation.
- Tags: RAG · SFT · GRPO · Model distillation · Medical QA
- Github repo: https://github.com/summer-royal/224r-medical_reasoning.git
- Paper (PDF): https://www.summer-royal.com/memory-as-action-cs224r.pdf

### Memory Distillation
- Description: Trains small language models to know when to distrust their own retrieved memory, using supervised fine-tuning and curriculum learning over teacher-generated distillations of a retrieval corpus. The best curriculum moved ProtocolQA 31 points and LitQA2 5 points over the 7B zero-shot baseline on LAB-Bench. It also has a limit worth stating out loud: targeted surface-form attacks defeat both similarity-based and semantic write-gate defenses, because they exploit the exact signal retrieval depends on.
- Tags: SFT · Curriculum learning · LAB-Bench · Adversarial robustness
- Github repo: https://github.com/rrsudev/dc-coscientist.git
- Paper (PDF): https://www.summer-royal.com/memory-distillation-cs224n.pdf

### Biomarkers from Slides
- Description: Asks whether attention-based multiple-instance learning can read ER, PR, and HER2 status off H&E-stained whole-slide images — the cheap stain every case already gets — instead of the assays that gate cancer treatment eligibility. Benchmarked three patch encoders (ResNet-50, UNI, CONCH) against three aggregation strategies, including a proposed Tumor-Aware CLAM with residual gating. Patient-level 5-fold cross-validation with bootstrap confidence intervals returned the unglamorous answer: dataset size is the binding constraint, not the architecture.
- Tags: Attention MIL · CLAM · UNI · CONCH · Digital pathology
- Github repo: https://github.com/summer-royal/cs231n_predicting_cancer_biomarkers.git
- Paper (PDF): https://www.summer-royal.com/biomarkers-from-slides-cs231n.pdf

### Exploding Kittens Agents
- Description: A full-fidelity simulation of the card game — roughly 100,000 hashed states, no simplifying assumptions — and three agents competing inside it: MLE with value iteration, Q-Learning with temporal-difference updates, and a Bayesian agent doing Dirichlet-Beta inference over deck composition. Q-Learning led at a 7.4% win rate over 500 games and 27% in all-agent tournaments, against a 25% random baseline. The better result is the explanation for that ceiling: the game's ~9% per-turn draw risk caps how much advantage any policy can extract.
- Tags: Q-Learning · Value iteration · Bayesian inference · Python
- Github repo: https://github.com/summer-royal/cs238-explodingKittens.git
- Paper (PDF): https://www.summer-royal.com/exploding-kittens-cs238.pdf

### AutonomyAid
- Description: A web-based platform, first built in 36 hours at TreeHacks at Stanford, that helps older adults document and enforce their own end-of-life care decisions. It combines scheduled execution of DNRs and advance directives with automatic upload to the patient's electronic health record, free customizable legal templates, and plain-language ethical education built around real cases. Built for the browser rather than mobile, since much of this population has computer access but no smartphone, and since arthritis and degenerative vision conditions make small touch targets a real barrier. Podcasts, a book club, and games address the link between social isolation and cognitive decline — and turn a one-time paperwork task into a reason to return.
- Tags: Web app · Accessibility · EHR integration · Advance care planning · Medical ethics
- Paper (PDF): https://www.summer-royal.com/autonomyaid.pdf
- Repository Link: https://github.com/miralomaadam/treehacks2024

### HipTracks
- Description: An iOS app that watches hip and knee replacement patients recover, pairing Apple Watch biometrics with a convolutional classifier (built with transfer learning) that reads wound photographs and predicts surgical site infection risk. Infections after joint replacement get caught at the follow-up appointment, which is often days later than the wound and the vitals first drift. Led the front end in Swift and wired the backend through CardinalKit and Apple HealthKit. Built in 48 hours at health{hacks} at Stanford — 1st place in the Aging & Longevity category — and kept going after.
- Tags: Swift · CNN · Transfer learning · CardinalKit · HealthKit
- Slides: https://docs.google.com/presentation/d/1OLkFK-Cn6qybLkbf03aS-T2s-viq9hBNgNWC3rOWqIg/view
- Repository Link: https://github.com/summer-royal/hipTracks_app.git

### NeuroTrack
- Description: A hardware-plus-software system that helps neurologists track disease progression through repeated reaction-time measurement, with a focus on Parkinson's, where early detection and monitoring between visits are major gaps. An Arduino Giga R1 rig (two buttons, two LEDs) runs a ten-trial reaction sequence capturing press latency and hold duration, feeding a web platform where clinicians manage patients, run tests, and view longitudinal charts against healthy and Parkinson's baselines. I worked on the classification layer, comparing Naive Bayes against a RandomForest classifier trained on reaction-time results plus medical history, labs, and symptoms; the RandomForest reached 86% accuracy.
- Tags: Arduino · Naive Bayes · Random forest · Web app · Parkinson's disease
- Github repo: https://github.com/radiaw/hackmit24.git
- Slides (PDF): https://www.summer-royal.com/neurotrack-pitch.pdf

### Mind-Controlled Lightbulb
- Description: A lightbulb switched by thought alone, built through Stanford's Brain-Computer Interface club: an EEG electrode reads the wearer, and the software decides when they meant it. Wrote the data collection and signal processing halves — the part that has to get from a noisy scalp electrode to a decision clean enough to act on.
- https://github.com/summer-royal/telecontrol.git
- Tags: EEG · Signal processing · Python · BCI

### Climate Mind
- Description: An app that lets people explore how the things they personally value are being affected by climate change. Trained the model underneath it: it processes incoming news articles continuously, filters for the factual content, and pulls out the cause-and-effect relationships. The longer aim is for something like it to run inside social platforms and mark false information where people actually meet it.
- Tags: NLP · Causal extraction · Python
- Website: https://climatemind.org/

### Foster Tower Tree
- Description: An app built during the COVID pandemic for the residents of a Honolulu condominium tower, so a neighbor who could not safely leave the building could ask the neighbors who could — shopping runs, rides to the doctor, a load of laundry.
- Tags: App development · Community software

### Ditch Dat!
- Description: A novel pediculicide — a head lice treatment that is eco-friendly, affordable, and patent-pending. First place at the Hawaii State Science Fair, and federal government funding to prototype it further.
- Tags: Formulation · Product design · Patent-pending
- Press: [Honolulu Star-Advertiser article](https://www.staradvertiser.com/2019/09/06/hawaii-news/lee-cataluna/cataluna-what-this-girl-did-in-the-name-of-science/)

## Coding Samples

### Neural Network with Backpropagation Algorithm for XOR Problem

- Description: This C++ program implements a neural network with backpropagation algorithm to solve the XOR problem. The XOR problem is a classic problem in neural network literature, where the goal is to predict the output of an XOR gate given two binary inputs.
- Repository Link: https://github.com/summer-royal/neural_network_for_xor.git

### Improving Logistic Regression for Credit Risk Prediction
- Description: This code implements a neural network model for predicting recidivism using the COMPAS dataset. The model is trained and evaluated on the dataset, and hyperparameter tuning is performed to find the best-performing model.
- Respository Link: https://github.com/summer-royal/logistic_regression_loan_approval_biases.git

### Deep Learning for Recidivism Prediction
- Description: This python program implements a neural network model for predicting recidivism using the COMPAS dataset. The model is trained and evaluated on the dataset, and hyperparameter tuning is performed to find the best-performing model.
- Repository Link: https://github.com/summer-royal/deep_learning_compas_recidivism_score.git

### Assessing Biases in Logistic Regression Model for Gender Prediction

- Description: This code implements logistic regression using word embeddings for gender prediction. It uses pre-trained word vectors from the GloVe word embedding model.
- Repository Link: https://github.com/summer-royal/logistic_regression_gender_bias.git

### Local Maxima Analysis of Neuronal Firing Rates
- Description: This code analyzes the firing rates of a neuron and identifies the local maxima in the rate data. It plots the firing rate over time and marks the local maxima points on the graph.
- Repository Link: https://github.com/summer-royal/local_max_firing_rate.git

## Skills

- Programming languages: C, C++, Python, HTML, Java, and Swift
- I have experience working with data structures, website development, app development, machine learning, deep learning, and data mining.

## Education

- M.S. Computer Science, AI Concentration | Stanford University | 2026
- B.S. Biomedical Computation | Stanford University | 2025

## Contact Information
www.summer-royal.com
https://www.linkedin.com/in/summer-royal-7824b5211
summerroyal@alumni.stanford.edu
