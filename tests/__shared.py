import os

# Folders
testDir = os.path.dirname(__file__) + '/'
packageDir = os.path.dirname(os.path.dirname(__file__)) + '/'

# Input files
inputDir = testDir + 'inp/'
snapshotFile = inputDir + 'snap_Gadget_sample'
spectrumApec = inputDir + 'apec_fakeit_for_test.pha'
spectrumBapec = inputDir + 'bapec_fakeit_for_test.pha'
spectrumApecNoStat = inputDir + 'apec_fakeit_nostat_for_test.pha'
spectrumBapecNoStat = inputDir + 'bapec_fakeit_nostat_for_test.pha'

# Reference files
referenceDir = testDir + 'reference_files/'
referenceSpecTableFile = referenceDir + 'reference_emission_table.fits'
referenceSpmapFile = referenceDir + 'reference.spmap'
referenceSimputFile = referenceDir + 'reference.simput'
referenceEvtFile = referenceDir + 'reference.evt'
referencePhaFile = referenceDir + 'reference.pha'
referenceSpfFile = referenceDir + 'reference.spf'
referenceErositaSimputFile = referenceDir + 'reference_erosita.simput'
referenceErositaPointedEvtFile = referenceDir + 'reference_erosita_pointed.evt'
referenceErositaPointedPhaFile = referenceDir + 'reference_erosita_pointed.pha'
referenceErositaGTIFile = referenceDir + 'reference_erosita_survey.gti'
referenceErositaSurveyEvtFile = referenceDir + 'reference_erosita_survey.evt'
referenceErositaSurveyPhaFile = referenceDir + 'reference_erosita_survey.pha'
specFitReferenceFile = referenceDir + 'reference_bapec_wrong_pars.spf'

# Test instruments
testInstrumentName = 'xrism-resolve-test'
testErositaPointedName = 'erosita-test'
testErositaSurveyName = 'erass1-test'

def clear_file(file: str) -> None:
    if os.path.exists(file):
        os.remove(file)
