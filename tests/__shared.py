import os

# Folders
path = os.path.dirname(__file__) + '/'

# Input files
inputDir = path + 'inp/'
snapshotFile = inputDir + 'snap_Gadget_sample'
spectrumApec = inputDir + 'apec_fakeit_for_test.pha'
spectrumBapec = inputDir + 'bapec_fakeit_for_test.pha'
spectrumApecNoStat = inputDir + 'apec_fakeit_nostat_for_test.pha'
spectrumBapecNoStat = inputDir + 'bapec_fakeit_nostat_for_test.pha'

# Reference files
referenceDir = path + 'reference_files/'
referenceSpecTableFile = referenceDir + 'reference_emission_table.fits'
referenceSpcubeFile = referenceDir + 'reference.speccube'
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



def clear_file(file: str) -> None:
    if os.path.exists(file):
        os.remove(file)
