\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{CPIPC}[2022/10/01 Custom Class]

\LoadClass[11pt]{ctexart}
\pagestyle{plain}


\RequirePackage{titlesec}
\RequirePackage{titletoc}
%\RequirePackage{hyperref} %目录和交叉引用超链接
\RequirePackage{ctex}
\RequirePackage{xeCJK}
\setCJKmainfont[AutoFakeBold=2.17]{FandolSong-Regular.otf}

\setCJKfamilyfont{zhli}{LiSu.ttf}
%\NewDocumentCommand \lishu{}{\CJKfamily{zhli}}
%\setCJKfamilyfont{timu}{heiti.ttf}
\setCJKfamilyfont{xw}{STXinwei.ttf}[AutoFakeBold=2.17]
\NewDocumentCommand \xinwei{}{\CJKfamily{xw}}

\ctexset{
    appendix        = {
        name        = {附录}
    },
    section         = {
        format      = {\bfseries\zihao{4}\heiti\centering},
        number      = {\bfseries\arabic{section}.},
        aftername   = {\hspace{0.5em}}
    },
    subsection      = {
        format      = {\bfseries\zihao{-4}},
        number      = {\bfseries\arabic{section}.\bfseries\arabic{subsection}},
        aftername   = {\hspace{0.5em}}
    },
    subsubsection   = {
        format      = {\bfseries\zihao{-4}},
        number      = {\bfseries\arabic{section}.\bfseries\arabic{subsection}.\bfseries\arabic{subsubsection}},
        aftername   = {\hspace{0.5em}}
    },
}
\titlecontents{section}[0pt]{\vspace{2mm}\bfseries\heiti}
{\thecontentslabel\hskip.5em}{}{\titlerule*[0.5pc]{.}\contentspage}
% 小节的目录格式
\titlecontents{subsection}[30pt]{}
{\thecontentslabel\hskip.5em}{}{\titlerule*[0.5pc]{.}\contentspage}

\titlecontents{subsubsection}[55pt]{}
{\thecontentslabel\hskip.5em}{}{\titlerule*[0.5pc]{.}\contentspage}

\pagestyle{plain}

    


  