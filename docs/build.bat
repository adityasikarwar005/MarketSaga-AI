@echo off
REM ---------------------------------------------------------------------
REM  MarketSage AI documentation build
REM
REM    tex\    LaTeX sources          (edit these)
REM    pdf\    compiled PDFs          (output)
REM    build\  aux / toc / log files  (scratch, safe to delete)
REM
REM  Two passes per document: pass 1 writes the .aux / .toc,
REM  pass 2 resolves the table of contents and cross-references.
REM  The master must build FIRST -- the part PDFs read its .aux via xr
REM  so their cross-references into the full document resolve.
REM ---------------------------------------------------------------------
setlocal
cd /d "%~dp0tex"

if not exist "..\pdf"   mkdir "..\pdf"
if not exist "..\build" mkdir "..\build"

set OPTS=-interaction=nonstopmode -output-directory=..\pdf -aux-directory=..\build

echo === master ===
pdflatex %OPTS% marketsage_master_plan.tex >nul
pdflatex %OPTS% marketsage_master_plan.tex >nul
if errorlevel 1 echo   ^!^! master FAILED - see build\marketsage_master_plan.log

for %%P in (04 05 06 07 08 09 10) do (
  echo === part%%P ===
  pdflatex %OPTS% part%%P.tex >nul
  pdflatex %OPTS% part%%P.tex >nul
  if errorlevel 1 echo   ^!^! part%%P FAILED - see build\part%%P.log
)

echo.
echo Done. PDFs are in pdf\
endlocal
