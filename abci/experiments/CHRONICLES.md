# Logbook

## May 16, 2023 - 11:00am (JST)
- (11am) Initiated training. There was a minor delay in the launch.
- (12pm) Terminated the job due to excessive checkpoint saves. The parameter `--eval_iters` was adjusted from 250 to 1000 to address this issue.
- (12pm) Training stalled after 1,500 iterations, potentially caused by the modification of `--eval_iters` during the training process. The exact cause remains unidentified. The decision was made to restart the training from the beginning.
