---
layout: post
title: "[HeatIsland] 도시 데이터 시각화 샘플"
author: "JeonghyunGan"
categories: project
tags: [heatisland, r]
comments: true
---

R `rayshader`를 활용한 도시 데이터 시각화 샘플

## 0. Summary

![rayshader](/assets/img/docs/rayshader.gif)

**현재는 실시간으로 받아와 처리할 수 있는 도시 데이터가 존재하지 않기 때문에, R ggmap에서 가져온 서울시 지도와 랜샛 위성사진을 활용하여 서울시의 열섬현상을 시각화하였다. 도시 내 IoT센서를 통해 실시간으로 데이터를 수집하고 처리하게 된다면 입력 데이터를 10분 단위로 업데이트하는 식으로 실시간 반영 가능할 것으로 예상된다.**

### 1. 데이터 전처리

먼저 랜샛 데이터를 읽어오고 서울시 경계에 맞게 잘라낸다.

```r
seoul = rgdal::readOGR("seoul/seoulExtent.shp")

band4 = raster::raster("seoul/20150704/LC08_L1TP_116034_20150704_20170407_01_T1_B4.TIF") %>%
  crop(extent(seoul))

band5 = raster::raster("seoul/20150704/LC08_L1TP_116034_20150704_20170407_01_T1_B5.TIF") %>%
  crop(extent(seoul))

band10 = raster::raster("seoul/20150704/LC08_L1TP_116034_20150704_20170407_01_T1_B10.TIF") %>%
  crop(extent(seoul))
```

다음으로 데이터로부터 밝기온도를 계산한다.

```r
getLST = function(B4, B5, B10){
  Rad = 0.0003342*B10+0.1
  Kelvin = 1321.08/log((774.89/Rad)+1)
  BT = Kelvin - 273.15
  NDVI = (B5-B4)/(B5+B4)
  Pv = ((NDVI-min(getValues(NDVI)))/(max(getValues(NDVI))/min(getValues(NDVI))))^2
  Emissivity = 0.004 * Pv + 0.986
  LST = (BT/(1+(0.00115*BT/1.4388) * log(Emissivity)))
  return(LST)
}

LST = getLST(band4, band5, band10)
```

LST를 계산한 후 지표온도 33도 이상과 미만으로 재코딩한다. 이 결과를 png 형식의 RGB 데이터로 변환하여 저장한다.

```r
values(LST) = case_when(
  values(LST)<33~ 0,
  values(LST)>=33 ~ 1
  )

LST %>% raster::RGB(col=c("#FFFFFF", RColorBrewer::brewer.pal(9, 'Reds')[6])) %>%
  as("SpatialGridDataFrame") %>%
  rgdal::writeGDAL("seoul/LST.png",
                 drivername = "PNG")

LST = png::readPNG("seoul/LST.png")
```

## 3. 고도 데이터 전처리

[JAXA](https://www.eorc.jaxa.jp/ALOS/en/aw3d30/)에서는 global elevation data를 제공한다. 한국 경위도에 맞는 데이터를 다운로드한 후  ``projectRaster`` 함수를 이용해 좌표계와 해상도(resolution)를 랜샛 데이터에 맞추어 다시 투영한다. 이 데이터를 서울시 경계에 맞게 자르고 ``rayshader``에서 사용 가능한 매트릭스 형태로 변환하였다.

```r
elev_img = raster::raster("seoul/seoulElev.tif")

elev_matrix = matrix(
  raster::extract(elev_img, extent(elev_img), buffer = 1000),
  nrow = ncol(elev_img), ncol = nrow(elev_img)
)

rm(elev_img)
```

## 3. 서울 지도 전처리

```r
gmap <- geocode(enc2utf8("서울")) %>%
  as.numeric() %>%
  get_googlemap(zoom=11,
                maptype='roadmap',
                format="png8")

mat = as.matrix(gmap)
vec = as.vector(mat)
vec_rgb = col2rgb(vec)
gmapR = matrix(vec_rgb[1, ], ncol = ncol(mat), nrow = nrow(mat))
gmapG = matrix(vec_rgb[2, ], ncol = ncol(mat), nrow = nrow(mat))
gmapB = matrix(vec_rgb[3, ], ncol = ncol(mat), nrow = nrow(mat))
roadmap = brick(raster(gmapR), raster(gmapG), raster(gmapB))
projection(roadmap) <- CRS("+init=epsg:3857")

LonLat = roadmap %>%
  raster::projectRaster(crs="+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0")

bb = attr(gmap, "bb")
gmap_extent = extent(bb$ll.lon, bb$ur.lon, bb$ll.lat, bb$ur.lat)
extent(LonLat) = gmap_extent

LonLat %>%
  raster::projectRaster(crs="+proj=utm +zone=52 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0") %>%
  crop(extent(seoul)) %>%
  as("SpatialGridDataFrame") %>%
  rgdal::writeGDAL("seoul/roadmap.png", drivername = "PNG")

rm(gmapR, gmapG, gmapB, mat, vec, vec_rgb, LonLat, gmap_extent, bb)
```

먼저 래스터 데이터를 위경도에 맞추어 투영하고, 올바른 extent를 지정해준다. 다음으로 랜샛 위성과 동일한 utm+로 다시 투영하고 서울시 경계에 맞게 잘라낸다. 처리한 데이터를 png이미지로 저장한다. 이후 png파일 크기를 1234 X 1012로 조정하였다.

```r
roadmap = png::readPNG("seoul/roadmap_resized.png")
```

## 4. ``rayshader``를 이용한 시각화

```r
ambmat = ambient_shade(elev_matrix, zscale = 30)
raymat = ray_shade(elev_matrix, zscale = 30, lmbert = TRUE)
watermap = detect_water(elev_matrix)
zscale = 10

rgl::clear3d()

elev_matrix %>%
  sphere_shade(texture = "imhof4") %>%
  add_water(watermap, color = "imhof4") %>%
  add_shadow(raymat, max_darken = 0.5) %>%
  add_shadow(ambmat, max_darken = 0.5) %>%
  add_overlay(LST, alphacolor = 0.5, alphalayer = 0.9) %>%
  add_overlay(roadmap, alphacolor = 0.5, alphalayer = 0.4) %>%
  plot_3d(elev_matrix,
          zscale=zscale,
          windowsize = c(900, 600),
          water=TRUE,
          soliddepth = -max(elev_matrix)/zscale,
          wateralpha = 0,
          theta = 25,
          phi = 30,
          zoom = 0.65,
          fov = 60)

render_movie("sample")
```
