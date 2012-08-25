#r "Microsoft.VisualBasic"
open Microsoft.VisualBasic.FileIO
#load "Data.fs"
open Charon.Data

let questionsFile = @"Z:\Data\StackOverflow\train-sample\train-sample.csv"
let parsed = parseQuestions questionsFile