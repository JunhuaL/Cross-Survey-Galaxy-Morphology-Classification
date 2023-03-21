#!/bin/bash
python baselineResnet.py Galaxy10.h5 False None
python baselineResnet.py Galaxy10.h5 True None
python baselineResnet.py Galaxy10.h5 True 30
python baselineResnet.py Galaxy10.h5 True 3
