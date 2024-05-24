module ML.ANN.Embedding(Embedding(..), embedSymbols, unembedSymbols) where

import Data.Array.Accelerate as A
import Data.List
import Data.List.Split
import Data.Map
import Data.Maybe
import Data.Random.Normal
import Data.Tree
import Prelude as P
import System.Random

type Embedding a = Data.Map.Map a (Vector Double)

embedSymbols :: P.Ord a => [a] -> StdGen -> Int -> Embedding a 
embedSymbols list randomSeed numDimensions = do
    let norms = normals randomSeed
        norms2 = P.map (\x -> x * (sqrt (2.0 / (P.fromIntegral numDimensions :: Double)))) norms
        ds = chunksOf numDimensions norms2
        fn (word, vect) m = Data.Map.insert word (A.fromList (Z:.numDimensions) vect) m
    P.foldr fn Data.Map.empty (P.zip list ds)
        
unembedSymbols :: P.Ord a => [a] -> Embedding a -> [Vector Double]
unembedSymbols [] _ = []
unembedSymbols (head : tail) embedding | isNothing (Data.Map.lookup head embedding) = unembedSymbols tail embedding
unembedSymbols (head : tail) embedding = do 
    let (Just l) = Data.Map.lookup head embedding
    l : (unembedSymbols tail embedding)

treeify :: P.Ord a => [[a]] -> [Tree (Maybe a)]
treeify list = do
    let sorted = P.reverse (Data.List.sort list)
    intern sorted where
        intern :: P.Ord a => [[a]] -> [Tree (Maybe a)]
        intern [] = []
        intern ( head : rest) = do
            let headTree = treeifyList head
                restTree = intern rest
            mergeTrees (headTree : restTree)

        mergeTrees :: P.Eq a => [Tree (Maybe a)] -> [Tree (Maybe a)] 
        mergeTrees [] = []
        mergeTrees ((Node {rootLabel = a, subForest=s1}) : treeList) = do
            let filteredIn = P.foldr (P.++) [] (P.map (\(Node { subForest = s2}) -> s2) (P.filter (\(Node {rootLabel = b}) -> a P.== b) treeList))
                filteredOut = P.filter (\(Node {rootLabel = b}) -> a P./= b) treeList
                filteredIn2 = P.foldr (P.++) [] (P.map (\(Node { subForest = x} ) -> if (P.length x) P.== 0 then [Node {rootLabel = Nothing, subForest=[]}] else x) filteredIn)
            ( Node {rootLabel = a, subForest = (mergeTrees (s1 P.++ filteredIn))}) : (mergeTrees filteredOut)
                
        treeifyList :: [a] -> Tree (Maybe a)
        treeifyList [x] = Node {rootLabel = (Just x), subForest = [(Node {rootLabel = Nothing, subForest=[]})]}
        treeifyList ( head : tail) = Node {rootLabel = (Just head), subForest = [(treeifyList tail)]}
