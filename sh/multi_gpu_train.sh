export CONFIG='/root/myWorkBase/code/autoposter_ocr0.1/customConfigs/psenet_r50_fpnf.py'
export GPU_NUM=2
export WORK_DIR='/root/myWorkBase/code/autoposter_ocr0.1/workDir/on_non_paste/baseline'
# export CHECKPOINT_FILE='/root/myWorkBase/code/mmsegm_0.16/workDir/portrait/swin_upernet/exp2/latest.pth'
../tools/dist_train.sh  ${CONFIG}  ${WORK_DIR}  ${GPU_NUM}  #--resume-from ${CHECKPOINT_FILE} #--load-from ${CHECKPOINT_FILE} 

