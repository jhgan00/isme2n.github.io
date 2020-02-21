---
layout: post
title: "[HeatIsland] 프로젝트 계획서"
categories: project
tags: heatisland
comments: true
---

2019-2학기에 진행된 '데이터사이언스입문' 수업에서 진행한 프로젝트의 연장선이다. 프로젝트 주제는 '서울시 열섬현상 완화를 위한 녹지 및 바람길 입지 선정'이었고, 관련 내용은 [깃허브 저장소](https://github.com/shd04121/heat_island_ds_yonsei)에서 확인할 수 있다. 해당 프로젝트에서는 랜샛 위성 데이터를 통해 서울시에서 열섬현상이 심한 곳을 선별하고 해결책을 제시하였다. 이번 프로젝트에서는 우선 70년대 이후부터 존재하는 랜샛 데이터를 통해 서울시 지표온도를 연도별/월별로 시각화하는 것을 목적으로 한다.  `raster`, `rayshader` 등의 R 패키지 위주로 지리 데이터를 다룬다.

## 1. 목표, 자원, 역할

 1) 목표 : 1970년 이후 한국의 여름철 온도 분포를 시각화한다.  

 프로젝트의 목표는 서울, 경기, 전주 지역을 중심으로 1970년 이후 한국의 여름철 온도 분포를 시각화하는 것이다. 모든 연도에 기준을 충족하는 데이터가 존재하지는 않으므로 약 5년 단위로 끊어서 작업한다. 최종적인 결과물은 각 지역별, 연도별로 시각화 결과를 확인할 수 있는 R shiny app으로 산출한다.

 2) 자원 : 팀원 5명, 오픈소스 언어 R, Python과 기타 패키지들

 3) 역할 : 기본적으로 모든 팀원이 모든 작업 단계에 참여하지만, 각 팀원이 모든 분야를 공부하여 작업하는 방식은 효율적이지 않으므로 작업 단계별로 한 명이 주도하여 진행한다.

  - 수집 : 아무나
  - 처리 및 시각화 : 아무나
  - 샤이니 개발 : 아무나

## 2. 데이터  

### 1) [Landsat 데이터](https://cloud.google.com/storage/docs/public-datasets/landsat?hl=ko)


> Landsat은 인공 위성을 사용하여 지구를 지속적으로 관찰하기 위한 목적에 따라 USGS와 NASA가 공동으로 개발한 프로그램입니다. Landsat 프로그램은 지표면을 우주에서 가장 오랫동안 지속적으로 기록한 대장정으로서, 1972년 Landsat 1 위성부터 시작되었습니다. Landsat 4부터 시작하여 각 위성은 다중 스펙트럼 및 열 계측기를 사용하여 지표면을 2주에 한 번씩 30미터 해상도로 촬영했습니다.

- Landsat 4: 1982 - 1993
- Landsat 5: 1984 - 2013
- Landsat 7: 1999 - 현재
- Landsat 8: 2013 - 현재


### 2) [JAXA ALOS Global Digital Surface Model 데이터](https://www.eorc.jaxa.jp/ALOS/en/aw3d30/)


>This data set is a global digital surface model (DSM) with horizontal resolution of approximately 30 meters (1 arcsecond) by the Panchromatic Remote-sensing Instrument for Stereo Mapping (PRISM) on board the Advanced Land Observing Satellite "ALOS".

|Dataset Details||
|------|-----|
|Product|ALOS World 3D – 30m (AW3D30) Version 2.2|
|Resolution|1 arcsecond (approximately 30 meters mesh)|
|File Unit|1 × 1 degree Latitude/Longitude Tile|
|File Composition|DSM file: Height above sea level (average value of source AW3D is adopted)|


## 3. 도구  

 1. R과 지리 데이터 관련 패키지   

  - sf : 벡터 데이터 처리
  - raster : raster 데이터 처리
  - rayshader : 3D 시각화
  - shiny : GUI 앱 개발

 2. Python


## 4. 일정  

|날짜|내용|
|----|----|
|7/20(토) ~ 7/26(금)|데이터 수집|
|7/27(토 ~ 8/16(금)|데이터 처리와 시각화|
|8/17(토) ~ 8/30(금)|샤이니 개발|