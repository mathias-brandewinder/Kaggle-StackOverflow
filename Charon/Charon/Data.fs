namespace Charon

module Data =

    open Microsoft.VisualBasic.FileIO

    let parseCsv (filePath: string) =

        let reader = new TextFieldParser(filePath)
        reader.TextFieldType <- FieldType.Delimited
        reader.SetDelimiters(",")

        Seq.unfold (fun line ->
            if reader.EndOfData 
            then 
                reader.Close()
                None
            else Some(line, reader.ReadFields())) (reader.ReadFields())