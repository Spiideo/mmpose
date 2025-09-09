# Guidelines for the Spiideo SockerNet SynLoc Challenge

We propose the SoccerNet challenges to encourage the development of state-of-the-art algorithm for Soccer Video Understanding.

 - **Spiideo SockerNet SynLoc Challenge**: Precise and accurate localisation of player on the pitch.

We provide an [evaluation server](https://www.codabench.org/competitions/10155/) for the Spiideo SockerNet SynLoc task.
The evaluation server handles predictions for the open **test** sets and the segregated **challenge** sets of each challenge.

Winners will be announced at CVSports Workshop at CVPR 2026.
This challenge will be sponsored by Spiideo, with a $1000 Amazon gift card prize to top participant!

## Evaluation metric
   The evaluation metric for the Spiideo SockerNet SynLoc task is the **mAP-LocSim** metric, a modified version of the standard mAP metric.
   A detailed description of this metric can be found in [the official paper](https://www.scitepress.org/publishedPapers/2025/131082/pdf/index.html).

## Submission format
   When running the baseline test script with the --chalenge option enabled, as described [here](https://github.com/Spiideo/mmpose/tree/spiideo_scenes?tab=readme-ov-file#challenge)
   a submission .zip file will be automatically generated in the current directory, e.g. 'challenge_submission.zip'.
   The submission file is a zipped folder containing a json file with the results and one with some metadata.
   The prediction json format is described [here](https://github.com/Spiideo/sskit?tab=readme-ov-file#map-locsim-evaluation).

## Who can participate / How to participate?

 - Any individual can participate in the challenge, except the organizers.
 - The participants are recommended to form a team to participate.
 - Each team can have one or more members.
 - An individual/team can compete on all tasks.
 - An individual associated with multiple teams (for a given task) or a team with multiple accounts will be disqualified.
 - A participant can only use the provided data as input.
 - The use of private datasets is *not* allowed. Teams using any kind of custom datasets, including additionnal annotations will be disqualified.

## How to win / What is the prize?

 - The winner is the individual/team who reaches the highest performance on the **challenge** set.
 - The metric taken into consideration is the **mAP-LocSim**, a modified version of the mAP metric.
 - To be eligible for the prize, we require the individual/team to provide a short report describing the details of the methodology (CVPR format, max 2 pages).


## Important dates

Note that these dates are tentative and subject to changes if necessary.
- September 9: Open evaluation server on the test set.
- September 9: Open evaluation server on the challenge set.
- April 25: Close evaluation server.
- May 1: Report submission deadline.
- TBD: CVSports Workshop at CVPR 2026 (awards ceremony).

## Clarifications on data usage

**1. On the restriction of private datasets and additional annotations**

SoccerNet is designed to be a research-focused benchmark, where the primary goal is to compare algorithms on equal footing. This ensures that the focus remains on algorithmic innovation rather than data collection or annotation effort. Therefore:
* Any data used for training or evaluation must be publicly accessible to everyone to prevent unfair advantages.
* By prohibiting additional manual annotations (even on publicly available data), we aim to avoid creating disparities based on resources (e.g., time, budget, or manpower). This aligns with our commitment to open-source research and reproducibility.

**2. On cleaning or correcting existing data**

We recognize that publicly available datasets, including SoccerNet datasets, might have imperfections in their labels (around 5% usually). Cleaning or correcting these labels is allowed outside of the challenge period to ensure fairness:
* Participants can propose corrections or improvements to older labels before the challenge officially starts. Such changes will be reviewed and potentially integrated into future versions of SoccerNet. Label corrections can be submitted before or after the challenge for inclusion in future SoccerNet releases, ensuring a fair and consistent dataset during the competition.
* During the challenge, participants should not manually alter or annotate existing labels, as this introduces inconsistency and undermines the benchmark's fairness.
* Fully automated methods for label refinement or augmentation, however, are encouraged. These methods should be described in the technical report to ensure transparency and reproducibility.

**3. Defining “private datasets”**

A dataset is considered “private” if it is not publicly accessible to all participants under the same conditions. For example:
* Older SoccerNet data are not private, as they are available to everyone.
* However, manually modifying or adding annotations (e.g., bounding boxes or corrected labels) to older SoccerNet data during the challenge creates a disparity and would be considered "private" unless those modifications are shared with the community in advance.

**4. Creative use of public data**

We fully support leveraging older publicly available SoccerNet data in creative and automated ways, as long as:
* The process does not involve manual annotations.
* The methodology is clearly described and reproducible.
* For instance, if you develop an algorithm that derives additional features or labels (e.g., bounding boxes) from existing data, this aligns with the challenge's goals and is permitted.

**5. Data sharing timeline:**

To ensure fairness, we decided that any new data must be published or shared with all participants through Discord at least one month before the challenge deadline. This aligns with the CVsports workshop timeline and allows all teams to retrain their methods on equal footing.


For any further doubt or concern, please raise an issue in that repository, or contact us directly on [Discord](https://discord.gg/SM8uHj9mkP).
