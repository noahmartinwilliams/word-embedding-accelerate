module ML.ANN.Embedding where

import Data.Array.Accelerate as A
import Prelude as P
import System.Random
import Data.Random.Normal
import Data.Map
import Data.List.Split
import Data.Maybe

type Embedding a = Data.Map.Map a (Vector Double)

embedSymbols :: P.Ord a => [a] -> StdGen -> Int -> Embedding a 
embedSymbols list randomSeed numDimensions = do
    let norms = normals randomSeed
        norms2 = P.map (\x -> x * (sqrt (2.0 / (P.fromIntegral numDimensions :: Double)))) norms
        ds = chunksOf numDimensions norms2
        fn (word, vect) m = insert word (A.fromList (Z:.numDimensions) vect) m
    P.foldr fn Data.Map.empty (P.zip list ds)
        
unembedSymbols :: P.Ord a => [a] -> Embedding a -> [Vector Double]
unembedSymbols [] _ = []
unembedSymbols (head : tail) embedding | isNothing (Data.Map.lookup head embedding) = unembedSymbols tail embedding
unembedSymbols (head : tail) embedding = do 
    let (Just l) = Data.Map.lookup head embedding
    l : (unembedSymbols tail embedding)
