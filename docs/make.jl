using DBF
using Documenter

DocMeta.setdocmeta!(DBF, :DocTestSetup, :(using DBF); recursive=true)

makedocs(;
    modules=[DBF],
    authors="Nick Mayhall",
    sitename="DBF.jl",
    format=Documenter.HTML(;
        canonical="https://nmayhall.github.io/DBF.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/nmayhall/DBF.jl",
    devbranch="main",
)
