# Contributing to MeatVisionAI

First off, thank you for considering contributing to MeatVisionAI! It's people like you that make open source such a great community.

## Where do I go from here?

If you've noticed a bug or have a feature request, make one! It's generally best if you get confirmation of your bug or approval for your feature request this way before starting to code.

## Fork & create a branch

If this is something you think you can fix, then fork MeatVisionAI and create a branch with a descriptive name.

A good branch name would be (where issue #325 is the ticket you're working on):

```sh
git checkout -b 325-add-new-model-architecture
```

## Setup & Testing

Please ensure you have all requirements installed and that the tests pass before submitting a pull request.

```sh
pip install -r requirements.txt
pytest tests/
flake8 Backend Scripts
black Backend Scripts
```

## Commit your changes

Commit your changes with a descriptive commit message.

## Submit a Pull Request

At this point, you should switch back to your master branch, make sure it's up to date with MeatVisionAI's master branch, and then submit a pull request!
