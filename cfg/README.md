# Configuration Files

Put configuration files for experiments and environments. Put them in `.yaml`
format. The `.json` files are from the 184 class, and are there to help get
hyperparameters to match what the class did, so that we can get folds done
properly. **See the examples/README.md for additional documentation**.

For the sake of keeping experiments reproducible and to avoid extensive tuning,
use one `.yaml` file per example script we want to run, where the scripts may
require different parameters.

We don't follow OpenAI gym conventions exactly. Normal gym environments assume
exact stability in the environment definitions. But we want to pass in our
`.yaml` configurations.

Most hyperparameters are straightforward and hopefully documenting one of the
configuration files will be sufficient.

- `cloth_from_184.yaml` -- some values borrowed from 184. Only use for testing
  the cloth itself, not the environment.

- `defaults.yaml` -- some reasonable values that we can use as a starting point
  for a variety of settings.
