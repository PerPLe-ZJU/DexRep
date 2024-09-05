# Introduction
The repository is the official codes of DexRepNet++, which includes two manipulation tasks (**In-hand Reorientation** and **HandOver**). 

[//]: # (For grasping with MuJoCo, you can access the codes in the [repository.]&#40;&#41;)

For the **grasping** task with IsaacGym, you can access the codes in the [repository.](https://github.com/LQTS/DexRep_Isaac)
# Dependencies
- Create a conda environment 
    ```shell
    conda create -n your_env_name python==3.8
    conda activate your_env_name
    ```
- Install torch
    ```shell
    pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
    ```
- Install IsaacGym

    1. Download [isaacgym](https://developer.nvidia.com/isaac-gym/download) 
    2. Extract the downloaded files to the main directory of the project
    3. Use the following commands to install isaacgym  
  ```shell
    cd isaacgym/python
    pip install -e .
    ```

The above commands show how to install the major packages. You can install other packages by yourself if needed.

# Run the scripts
```Refer to the examples in the folder Train_Manipulation/script``` 
