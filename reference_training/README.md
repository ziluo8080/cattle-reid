# Historical Training Source Excerpts

These seven source files are byte-preserved excerpts recovered from the historical cloud runtime. Their hashes were checked against training release records. They document the actual network, three objectives, optimizer calls and learning-rate schedule.

They are **not a standalone training distribution**: internal package imports, data materialization contracts, historical launch authorizations and model weights are not bundled. Do not run them expecting a complete training workflow. No current cloud launch is authorized by this public archive.

For an executable reproduction of the archived numerical results, use `scripts/reproduce_results.py`. For the configuration and exact source hashes, see `protocols/training_config.json`. Model retraining and fresh image inference are outside this release.

`controlled_adapter_training.py` contains older adapter helpers as well as the shared learning-rate function. Only its `learning_rate` function is used by these DenseNet training entries. It must not be read as evidence that the present experiment uses the adapter architecture.
