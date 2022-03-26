# -*- coding:utf-8 -*-
import os


class SOV():
    def runSOV(self, fastapath, outpath):
        names = [name for name in os.listdir(fastapath) if os.path.isfile(os.path.join(fastapath + '//', name))]
        for each_item in names:
            pdb_id = each_item.split('.')[0]
            '''
            https://github.com/nanjiangshu/calSOV
            g++ calSOV.cpp
            ./calSOV/a.out -f 2 test/test1_f2.txt
            '''
            cmd = './calSOV/a.out -f 2 ' + fastapath + '/' + each_item + ' -o ' + outpath + '/' + pdb_id + '.txt '
            os.system(cmd)


if __name__ == '__main__':
    '''
    samples:
    fastapath = './calSOV/test/win11test/hmmlstm_sov'
    outpath = './calSOV/test/win11test/hmmlstm_sov_result'
    You can also check the structure and format of used fasta files in my folder : ./calSOV/test/win11test/hmmlstm_sov
    Warning : the permissions issue has not been resolved, please use the filepath under /home/RaidDisk/ as your outpath
    '''

    fastapath = "./calSOV/test/win11test/hmmlstm_sov"
    outpath = "./calSOV/test/win11test/hmmlstm_sov_result"

    sov = SOV()
    sov.runSOV(fastapath, outpath)

# tar zxvf /usr/share/hhsuite/database/uniprot20_2016_02.tgz -C /usr/share/hhsuite/database/
# cp /home/lznenu/hh-suite/databases/uniprot20_2016_02.tgz /usr/share/hhsuite/database/
# wget -P /home/third_part_tools/hhsuite/databases/ http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/old-releases/uniprot20_2016_02.tgz