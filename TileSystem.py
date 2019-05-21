#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 14:41:39 2019

@author: luoshiyu
"""
import re
from itertools import chain
import numpy as np

class TileSystem:
    earthRadius = 6378137
    minLatitude = -85.05112878
    maxLatitude = 85.05112878
    minLongitude = -180
    maxLongitude = 180
    
    MAXLEVEL = 23
    
    def __init__(self):
        pass

    @staticmethod
    def latLongToPixelXY(lats, lons, levelOfDetail):
        """
        Calculates the pixel coordinates of points specified by <lats> and <lons>
        in detail level <levelOfDetail>.
        Params:
            lats: 1-D numpy array, shape (N,), latitudes of the N points
            lons: 1-D numpy array, shape (N,), longitudes of the N points
            levelOfDetail: level of detail, from 1 (lowest) to 23 (highest).
        Returns:
            pixelXs: 1-D int numpy array, shape (N,), X coordinates in pixels.
            pixelYs: 1-D int numpy array, shape (N,), Y coordinates in pixels.
        """
        
        lats = np.clip(lats, TileSystem.minLatitude, TileSystem.maxLatitude)
        lons = np.clip(lons, TileSystem.minLongitude, TileSystem.maxLongitude)
        
        x = (lons + 180) / 360
        sinLatitudes = np.sin(lats * np.pi / 180)
        y = 0.5 - np.log((1 + sinLatitudes) / ( 1 - sinLatitudes)) / (4 * np.pi)
        
        mapSize = 256 << levelOfDetail
        pixelXs = np.clip(x * mapSize + 0.5, 0, mapSize - 1)
        pixelYs = np.clip(y * mapSize + 0.5, 0, mapSize - 1)
        
        return pixelXs.astype(int), pixelYs.astype(int)
    
    @staticmethod
    def latLongToQuadKey(lats, lons, levelOfDetail):
        pixelX, pixelY = TileSystem.latLongToPixelXY(lats, lons, levelOfDetail)
        tileX, tileY = TileSystem.pixelXYToTileXY(pixelX, pixelY)
        quadKey = TileSystem.tileXYToQuadKey(tileX, tileY, levelOfDetail)
        return quadKey
    
    @staticmethod
    def pixelXYToLatLong(pixelXs, pixelYs, levelOfDetail):
        mapSize = 256 << levelOfDetail
        x = np.clip(pixelXs, 0, mapSize - 1) / mapSize - 0.5
        y = 0.5 - np.clip(pixelYs, 0, mapSize - 1) / mapSize
        
        lats = 90 - 360 * np.arctan(np.exp(-y * 2 * np.pi)) / np.pi
        lons = 360 * x
        
        return lats, lons


    @staticmethod
    def pixelXYToTileXY(pixelXs, pixelYs):
        """
        Converts pixel XY coordinates into tile XY coordinates of the tiles
        containing the specified pixels.
        Params:
            pixelXs: 1-D int numpy array, shape (N,), pixel X coordinates.
            pixelYs: 1-D int numpy array, shape (N,), pixel Y coordinates.
        Returns:
            tileXs: 1-D int numpy array, shape (N,), tile X coordinates.
            tileYs: 1-D int numpy array, shape (N,), tile Y coordinates.
        """
        
        return pixelXs // 256, pixelYs // 256
    
    
    @staticmethod
    def tileXYToPixelXY(tileX, tileY):
        """
        Converts tile XY coordinates into pixel XY coordinates of the upper-left
        pixel of the specified tile.
        """
        return tileX * 256, tileY * 256
    
    @staticmethod
    def tileXYToLatLong(tileX, tileY, levelOfDetail):
        pixelX, pixelY = TileSystem.tileXYToPixelXY(tileX, tileY)
        return TileSystem.pixelXYToLatLong(pixelX, pixelY, levelOfDetail)
    
    @staticmethod
    def latLongToTileXY(lats, lons, level):
        quadKey = TileSystem.latLongToQuadKey(lats, lons, level)
        tileX, tileY = TileSystem.quadKeyToTileXY(quadKey)
        return tileX, tileY
    
    
    @staticmethod
    def tileXYToQuadKey(tileX, tileY, level):
        """
        Converts tile XY coordinates into a QuadKey at a specified level of detail.
        """
        
        tileXbits = '{0:0{1}b}'.format(tileX, level)
        tileYbits = '{0:0{1}b}'.format(tileY, level)
        
        quadkeybinary = ''.join(chain(*zip(tileYbits, tileXbits)))
        return ''.join([str(int(num, 2)) for num in re.findall('..?', quadkeybinary)])
    
    @staticmethod
    def quadKeyToTileXY(quadKey):
        quadkeybinary = ''.join(['{0:02b}'.format(int(num)) for num in quadKey])
        tileX, tileY = int(quadkeybinary[1::2], 2), int(quadkeybinary[::2], 2)
        return tileX, tileY
    
    @staticmethod
    def quadKeyToBoundingBoxLatLong(quadKey):
        tileX, tileY = TileSystem.quadKeyToTileXY(quadKey)
        levelOfDetail = len(quadKey)
        
        # upper-left pixel
        pixelX1, pixelY1 = TileSystem.tileXYToPixelXY(tileX, tileY)
        lat1, lon1 = TileSystem.pixelXYToLatLong(pixelX1, pixelY1, levelOfDetail)
        
        # lower-right pixel
        pixelX2, pixelY2 = TileSystem.tileXYToPixelXY(tileX + 1, tileY + 1)
        lat2, lon2 = TileSystem.pixelXYToLatLong(pixelX2, pixelY2, levelOfDetail)
        
        return lat1, lon1, lat2, lon2
    
    
    @staticmethod
    def quadKeyToFourCornersLatLong(quadKey):
        """
        Can be used to generate markups on google earth.
        """
        
        tileX, tileY = TileSystem.quadKeyToTileXY(quadKey)
        levelOfDetail = len(quadKey)
        
        # upper-left corner
        pixelX1, pixelY1 = TileSystem.tileXYToPixelXY(tileX, tileY)
        lat1, lon1 = TileSystem.pixelXYToLatLong(pixelX1, pixelY1, levelOfDetail)
        
        # upper-right corner
        lat2, lon2 = TileSystem.pixelXYToLatLong(pixelX1 + 255, pixelY1, levelOfDetail)

        # lower-left corner
        lat3, lon3 = TileSystem.pixelXYToLatLong(pixelX1, pixelY1 + 255, levelOfDetail)
        
        # lower-right corner
        lat4, lon4 = TileSystem.pixelXYToLatLong(pixelX1 + 255, pixelY1 + 255, levelOfDetail)
        
        return (lon1, lat1), (lon2, lat2), (lon3, lat3), (lon4, lat4)
        
        
    
    @staticmethod
    def boundingBoxToTileQuadKeys(lat1, lon1, lat2, lon2, levelOfDetail):
        pixelX1, pixelY1 = TileSystem.latLongToPixelXY(lat1, lon1, levelOfDetail)
        pixelX2, pixelY2 = TileSystem.latLongToPixelXY(lat2, lon2, levelOfDetail)
        
        tileX1, tileY1 = TileSystem.pixelXYToTileXY(pixelX1, pixelY1)
        tileX2, tileY2 = TileSystem.pixelXYToTileXY(pixelX2, pixelY2)
        
        fromX = min(tileX1, tileX2)
        toX = max(tileX1, tileX2)
        
        fromY = min(tileY1, tileY2)
        toY = max(tileY1, tileY2)
        
#        print('upper-left tile coordinate: (%d, %d)' % (fromX,  fromY))
#        print('lower-right tile coordinate: (%d, %d)' % (toX, toY))
        
        quadKeys = list()
        for tileX in range(fromX, toX + 1):
            for tileY in range(fromY, toY + 1):
                quadKey = TileSystem.tileXYToQuadKey(tileX, tileY, levelOfDetail)
                quadKeys.append(quadKey)
        
        return quadKeys
    
    def boundingBoxToTileQuadKeys2(lat1, lon1, lat2, lon2, levelOfDetail):
        pixelX1, pixelY1 = TileSystem.latLongToPixelXY(lat1, lon1, levelOfDetail)
        pixelX2, pixelY2 = TileSystem.latLongToPixelXY(lat2, lon2, levelOfDetail)
        
        tileX1, tileY1 = TileSystem.pixelXYToTileXY(pixelX1, pixelY1)
        tileX2, tileY2 = TileSystem.pixelXYToTileXY(pixelX2, pixelY2)
        
        fromX = min(tileX1, tileX2)
        toX = max(tileX1, tileX2)
        
        fromY = min(tileY1, tileY2)
        toY = max(tileY1, tileY2)
        
#        print('upper-left tile coordinate: (%d, %d)' % (fromX,  fromY))
#        print('lower-right tile coordinate: (%d, %d)' % (toX, toY))
        
        quadKeys = list()
        for tileY in range(fromY, toY + 1):
            rowQuadKeys = list()
            for tileX in range(fromX, toX + 1):
                quadKey = TileSystem.tileXYToQuadKey(tileX, tileY, levelOfDetail)
                rowQuadKeys.append(quadKey)
            quadKeys.append(rowQuadKeys)
        
        bbox1 = TileSystem.quadKeyToBoundingBoxLatLong(quadKeys[0][0])[:2]
        bbox2 = TileSystem.quadKeyToBoundingBoxLatLong(quadKeys[-1][-1])[2:]
        return quadKeys, bbox1, bbox2
        
        
    
            
        
        
        
    
    
    
        