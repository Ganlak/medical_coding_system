@echo off
REM Phase 2: CMS Data Pipeline
REM Usage: scripts\run_phase2.bat 2025

set YEAR=%1
if "%YEAR%"=="" set YEAR=2025

echo ================================================================================
echo PHASE 2: CMS DATA PIPELINE - YEAR %YEAR%
echo ================================================================================
echo.

REM Step 1: Fetch
echo STEP 1/3: FETCHING CMS CODES...
python ingestion/cms/fetch_cms_codes.py --year %YEAR% --codes all --output ./data/raw
if errorlevel 1 goto error
echo √ Fetch complete
echo.

REM Step 2: Normalize
echo STEP 2/3: NORMALIZING CMS CODES...
python ingestion/cms/normalize_cms_codes.py --year %YEAR% --codes all
if errorlevel 1 goto error
echo √ Normalize complete
echo.

REM Step 3: Embed
echo STEP 3/3: EMBEDDING CMS CODES...
python ingestion/embeddings/embed_cms_codes.py --year %YEAR% --codes all
if errorlevel 1 goto error
echo √ Embed complete
echo.

echo ================================================================================
echo √ PHASE 2 COMPLETE!
echo ================================================================================
goto end

:error
echo.
echo ================================================================================
echo X ERROR: Pipeline failed at step above
echo ================================================================================
exit /b 1

:end