namespace Charon

module Preprocessing =

    open System
    open System.IO
    open System.Text.RegularExpressions
    open Microsoft.VisualBasic.FileIO    

    let options = RegexOptions.Compiled ||| RegexOptions.IgnoreCase

    let codeRegex = Regex("((^|\\n)\s{4}\s*((?!\\n).)*)+|`[^`]+`", options)
    let httpAddrRegex = Regex("(http|https)://[^\s]*", options)

    let inline removeCode str = codeRegex.Replace(str, "")
    let inline removeLinks str = httpAddrRegex.Replace(str, "httpaddress")
    
    let preprocess str = (removeCode >> removeLinks) str