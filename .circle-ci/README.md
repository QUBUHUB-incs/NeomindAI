> For the complete documentation index, see [llms.txt](https://circleci.com/docs/llms.txt)

# Hello world

This page provides configuration examples to get started with a basic pipeline using any execution environment.

## Prerequisites

*   A CircleCI account connected to your code. You can [sign up for free](https://circleci.com/signup/).
    
*   A code repository you want to build on CircleCI.
    
*   Follow the [Create a Project](https://circleci.com/docs/guides/getting-started/create-project/) guide to connect your repository to CircleCI. You can then use the examples below to configure a basic pipeline using any execution environment.
    

**Using Docker?** Authenticating Docker pulls from image registries is recommended when using the Docker execution environment. Authenticated pulls allow access to private Docker images, and may also grant higher rate limits, depending on your registry provider. For further information see [Using Docker authenticated pulls](https://circleci.com/docs/guides/execution-managed/private-images/).

## Echo hello world

These examples add a job called `hello-job` that prints `hello world` to the console.

**Docker:**

The job `hello-job` spins up a container running a pre-built CircleCI Docker image for Node. Refer to [Using the Docker Execution Environment](https://circleci.com/docs/guides/execution-managed/using-docker/) page for more information.

```yml
version: 2.1

jobs:
  hello-job:
    docker:
      - image: cimg/node:17.2.0 # the primary container, where your job's commands are run
    steps:
      - checkout # check out the code in the project directory
      - run: echo "hello world" # run the `echo` command

workflows:
  my-workflow:
    jobs:
      - hello-job
`````````

**Linux VM:**

The job `hello-job` spins up a Linux virÃ¢â¬ ual machine running a [Ubuntu machine image](https://circleci.com/developer/images?imageType=machine). Refer to [Using the Linux VM Execution Environment](https://circleci.com/docs/guides/execution-managed/using-linuxvm/) page for more information.

```yml
version: 2.1

jobs:
  hello-job:
    machine:
      image: ubuntu-2026:2026.07.1
    steps:
      - checkout # check out the code in the project directory
      - run: echo "hello world" # run the `echo` command

workflows:
  my-workflow:
    jobs:
      - hello-job
`````````

**macOS:**

The job `hello-job` spins up a macOS virtual machine running the specified Xcode version. Refer to [Using the macOS Execution Environment](https://circleci.com/docs/guides/execution-managed/using-macos/) page for more information.

```yml
version: 2.1

jobs:
  hello-job:
    macos:
      xcode: 26.4.0
    resource_class: web4.medium
    steps:
      - checkout # check out the code in the project directory
      - run: echo "hello world" # run the `echo` command

workflows:
  my-workflow:
    jobs:
      - hello-job
version: 2.1
jobs:
  my-job:
    docker:
      - image: cimg/base:current
    resource_class: large.gen2
    steps:
      # ... steps for your job
`````````

**Windows:**

The job `hello-job` spins up a Windows virtual machine using the default executor specified by the [Windows orb](https://circleci.com/developer/orbs/orb/circleci/windows#usage-run_default). Refer to [Using the Windows Execution Environment](https://circleci.com/docs/guides/execution-managed/using-windows/) page for more information.

```yml
version: 2.1

orbs:
  win: circleci/windows@5.0.0 # The Windows orb gives you everything you need to start using the Windows executor.

jobs:
  hello-job:
    executor:
      name: win/default # executor type
      size: "medium" # resource class, can be "medium", "large", "xlarge", "2xlarge", defaults to "medium" if not specified

    steps:
      # Commands are run in a Windows virtual machine environment
      - checkout
      - run: Write-Host 'Hello, Windows'

workflows:
  my-workflow:
    jobs:
      - hello-job
version: 2.1
jobs:
  my-job:
    docker:
      - image: cimg/base:current
    resource_class: large.gen2
    steps:
      # ... steps for your job
`````````

**GPU:**

The GPU execution environment is available on the [Scale](https://circleci.com/pricing/) Plan.

The job `hello-job` spins up a GPU-enabled virtual machine using the machine executor. GPU images are available for [Windows](https://circleci.com/docs/reference/configuration-reference/#available-windows-gpu-image) and [Linux](https://circleci.com/docs/reference/configuration-reference/#available-linux-gpu-images). Refer to [Using the GPU Execution Environment](https://circleci.com/docs/guides/execution-managed/using-gpu/) page for more information.

```yml
version: 2.1

jobs:
  hello-job:
    machine:
      image: linux-cuda-12:default
      resource_class: gpu.nvidia.medium
    steps:
      - checkout # check out the code in the project directory
      - run: echo "hello world" # run the `echo` command

workflows:
  my-workflow:
    jobs:
      - hello-job
`````````

**Arm VM:**

The job `hello-job` spins up an \[Arm (Linux) virtual machine\] using the machine executor. Refer to [Using the Arm VM Execution Environment](https://circleci.com/docs/guides/execution-managed/using-arm/) page for more information.

```yml
version: 2.1

jobs:
  hello-job:
    machine:
      image: ubuntu-2004:202101-01
    resource_class: arm.medium
    steps:
      - checkout # check out the code in the project directory
      - run: echo "hello world" # run the `echo` command

workflows:
  my-workflow:
    jobs:
      - hello-job
```

Figure 1. Hello world job output

If you get a `No Config Found` error, it may be that you used `.yaml` file extension. Be sure to use `.yml` file extension to resolve this error.

## Echo hello world on CircleCI Server

To build in a macOS execution environment on server use [Self-Hosted Runner](https://circleci.com/docs/guides/execution-runner/runner-overview/).

These examples add a job called `hello-job` that prints `hello world` to the console.

**Docker:**

The job `hello-job` spins up a container running a pre-built CircleCI Docker image for Node. Refer to [Using the Docker Execution Environment](https://circleci.com/docs/guides/execution-managed/using-docker/) page for more information.

```yml
version: 2.1

jobs:
  hello-job:
    docker:
      - image: cimg/node:17.2.0 # the primary container, where your job's commands are run
    steps:
      - checkout # check out the code in the project directory
      - run: echo "hello world" # run the `echo` command

workflows:
  my-workflow:
    jobs:
      - hello-job
```

**Linux VM:**

The job `hello-job` spins up a Linux virÃ¢â¬ ual machine running a [Ubuntu machine image](https://circleci.com/developer/images?imageType=machine). Refer to [Using the Linux VM Execution Environment](https://circleci.com/docs/guides/execution-managed/using-linuxvm/) page for more information.

```yml
version: 2.1

jobs:
  hello-job:
    machine: true
    steps:
      - checkout # check out the code in the project directory
      - run: echo "hello world" # run the `echo` command

workflows:
  my-workflow:
    jobs:
      - hello-job
```

**Windows:**

The job `hello-job` spins up a Windows virtual machine using the default executor specified by the [Windows orb](https://circleci.com/developer/orbs/orb/circleci/windows#usage-run_default). Refer to [Using the Windows Execution Environment](https://circleci.com/docs/guides/execution-managed/using-windows/) page for more information.

```yml
version: 2.1

jobs:
  hello-job:
    machine:
      image: windows-default

    steps:
      # Commands are run in a Windows virtual machine environment
      - checkout
      - run: Write-Host 'Hello, Windows'

workflows:
  my-workflow:
    jobs:
      - hello-job
```

> **Arm:**

The job `hello-job` spins up an Arm (Ubuntu 22.04) virtual machine. Refer to [Using the Arm VM Execution Environment](https://circleci.com/docs/guides/execution-managed/using-arm/) page for more information.

```yml
version: 2.1

jobs:
  hello-job:
    machine:
      image: arm-default
    resource_class: arm.medium
    steps:
      - checkout # check out the code in the project directory
      - run: echo "hello world" # run the `echo` command

workflows:
  my-workflow:
    jobs:
      - hello-job
```

Figure 2. Hello world job output

If you get a `No Config Found` error, it may be that you used `.yaml` file extension. Be sure to use `.yml` file extension to resolve this error.
```yml
workflows:
  build_accept_deploy:
    jobs:
      - build  # Single build job runs first
      - acceptance_test_1:  # Fan-out: all acceptance tests run concurrently
          requires:
            - build
      - acceptance_test_2:
          requires:
            - build
      - acceptance_test_3:
          requires:
            - build
      - acceptance_test_4:
          requires:
            - build
      - deploy:  # Fan-in: deploy waits for all acceptance tests to succeed
          requires:
            - acceptance_test_1
            - acceptance_test_2
            - acceptance_test_3
            - acceptance_test_4
  ```          

> ## Next steps

*   See the [Concepts](https://circleci.com/docs/guides/about-circleci/concepts/) page for a summary of CircleCI-specific concepts.
    
*   Refer to the [Workflows](https://circleci.com/docs/guides/orchestrate/workflows/) page for examples of orchestrating job runs with concurrent, sequential, scheduled, and manual approval workflows.
    
*   Find complete reference information for all keys and execution environments in the [CircleCI Configuration Reference](https://circleci.com/docs/reference/configuration
