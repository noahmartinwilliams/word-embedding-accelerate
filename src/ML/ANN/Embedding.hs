module ML.ANN.Embedding(Embedding(..), embedSymbols, unembedSymbols, tokenize) where

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

{- Treeify will be used to build a binary tree whose branches indicate valid paths that a token can take.
   For example, if the input string starts with "abc" and the tree looks like:
   a
   |_b
   | |_c
   |_d
   | |_f
   |_e

   Then the begining of the input matches against the top node, then the b child node, then the c child node, and then its considered a valid token.
   This way we can make tokens that are as long as possible. So if the dictionary contains ["abc", "abcd"] it will greedily match the input "abcd".
-}

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
        treeifyList [x] = Node {rootLabel = (Just x), subForest = [(Node {rootLabel = Nothing, subForest=[]})]} --We'll use Nothings to indicate that a node on the tree is a valid end to a token, and an empty list represents invalid ends
        treeifyList ( head : tail) = Node {rootLabel = (Just head), subForest = [(treeifyList tail)]}

tokenize :: P.Ord a => [a] -> [[a]] -> [Either [a] [a]] -- Lefts will indicate a valid token, and Rights will indicate an invalid token. 
tokenize inputStream dictionary = do
    let tree = treeify dictionary
        eithers = intern [] inputStream tree tree 
    condense [] eithers where
        intern :: P.Ord a => [a] -> [a] -> [Tree (Maybe a)] -> [Tree (Maybe a)] -> [Either [a] a]
        intern token [] dictTree _ = do -- End of input stream.
            let filtered = P.filter (\(Node {rootLabel = x}) -> isNothing x) dictTree
            if (P.length filtered) P.== 0 then P.map (\x -> Right x) token else [Left token]
        intern tokenSoFar (head : tail) dictTree resetTree = do
            let filtered = P.filter (\(Node {rootLabel = x}) -> if (isJust x) P.&& (x P.== (Just head)) then True else False) dictTree
                filteredNothing = P.filter (\(Node {rootLabel=x}) -> if (isNothing x) then True else False ) dictTree
                tokenSoFar2 = tokenSoFar P.++ [head]
            if (P.length filtered) P.== 0 -- There are no paths down the tree that involve this character
            then 
                if (P.length filteredNothing) P./= 0 -- There is a nothing here, which is used to indicate that this is a valid end to the token
                then 
                    (Left tokenSoFar) : (intern [] (head : tail) resetTree resetTree) 
                else -- There is no Nothing here, so this isn't a valid end
                    if tokenSoFar P./= []  -- We have what a token that has been valid so far, but it's not long enough to match against any existing word in the dictionary
                    then
                        (P.map (\x -> Right x) tokenSoFar) P.++ (Right head) : (intern [] tail resetTree resetTree) 
                    else
                        (Right head) : (intern [] tail resetTree resetTree)
            else -- There is a path to go down further
                let filtered2 = filtered P.!! 0 in 
                    let (Node {subForest = subTree}) = filtered2 in
                        (intern tokenSoFar2 tail subTree resetTree)
                    
        condense :: [a] -> [Either [a] a] -> [Either [a] [a]]
        condense x [] | (P.length x) P./= 0  = [Right x]
        condense _ [] = []
        condense rightSoFar ((Left x) : rest) | (P.length rightSoFar) P./= 0 = (Right rightSoFar) : (Left x) : (condense [] rest)
        condense _ ((Left x) : rest) = (Left x) : (condense [] rest)
        condense rightSoFar ((Right x) : rest) = condense (rightSoFar P.++ [x]) rest
