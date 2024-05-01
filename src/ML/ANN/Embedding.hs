module ML.ANN.Embedding where

import Data.Array.Accelerate as A
import Prelude as P
import System.Random
import Data.Random.Normal
import Data.Map
import Data.List.Split

type Embedding a = Data.Map.Map a (Matrix Double)

words2Embedding :: P.Ord a => [a] -> StdGen -> Int -> Embedding a 
words2Embedding list randomSeed numDimensions = do
    let ds = chunksOf numDimensions (normals randomSeed)
        fn (word, vect) m = insert word (A.fromList (Z:.numDimensions:.1) vect) m
    P.foldr fn Data.Map.empty (P.zip list ds)
        
