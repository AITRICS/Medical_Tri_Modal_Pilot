import csv
from tqdm import tqdm

def refine_c_note(reader):
    refined_note = list()

    error_flag = False
    

    for row in tqdm(reader, desc='Refine C Note', ncols=75):
        if error_flag:
            if row['ChosNo'] == None:
                text = row['UnitNo']
                refined_note[error_idx]['RecVal'] += " {}".format(text)

            elif row['YMD'] == None:
                RgtDeptCd = row['UnitNo']
                RgtId = row['ChosNo']
                refined_note[error_idx]['RgtDeptCd'] = RgtDeptCd
                refined_note[error_idx]['RgtId'] = RgtId
                error_flag = False
            else:
                RgtDeptCd = row['ChosNo']
                RgtId = row['YMD']
                refined_note[error_idx]['RgtDeptCd'] = RgtDeptCd
                refined_note[error_idx]['RgtId'] = RgtId
                error_flag = False
        else:
            if row['UnitNo'].isdigit():
                if None in row:
                    row['RecVal'] += " {}".format(row['RgtDeptCd'])
                    row['RgtDeptCd'] = row['RgtId']
                    row['RgtId'] = row[None]
                    del row[None]

                refined_note.append(row)
                

            else:
                error_flag = True
                error_idx = len(refined_note) - 1
                text = row['UnitNo']
                refined_note[error_idx]['RecVal'] += " {}".format(text)



    return refined_note




def refine_i_note(reader):
    refined_note = list()

    error_flag = False
    

    for row in tqdm(reader, desc='Refine I Note', ncols=75):
        row['UnitNo'] = row['UnitNo'].replace('\n','').replace(',','').replace('"', '').strip()
        if row['UnitNo'].isdigit():
            if row['AttrCd'] in {'A0000653', 'A0013358', 'A0017215', 'A0017373'}:
                refined_note.append({'UnitNo': row['UnitNo'],
                                     'ChosNo': row['ChosNo'],
                                     'YMD': row['ymd'],
                                     'AttrCd': row['AttrCd'],
                                     'AttrNm': row['AttrNm'],
                                     'TEXT': row['AttrValue']})

    return refined_note
    

def main():
    NOTES = [
              '처치간호정보.csv',
              '입원기록.csv',
            ]

    # reader = csv.DictReader(open('처치간호정보.csv', 'r', encoding='utf-8-sig'))
    # header = reader.fieldnames
    # refined_c_note = refine_c_note(reader)
    # writer = csv.DictWriter(open('REFINED_NOTES/처치간호정보.csv', 'w', encoding='utf-8-sig'), fieldnames=header)
    # writer.writeheader()

    # for i, row in tqdm(enumerate(refined_c_note), desc='Write Refined C Note', ncols=75):
    #     writer.writerow(row)

    #     if not row['UnitNo'].isdigit():
    #         print(refined_c_note[i-1])
    #         print(row)

    reader = csv.DictReader(open('입원기록.csv', 'r', encoding='utf-8-sig'))
    refined_i_note = refine_i_note(reader)
    header = list(refined_i_note[0].keys())
    writer = csv.DictWriter(open('REFINED_NOTES/입원기록.csv', 'w', encoding='utf-8-sig'), fieldnames=header)
    writer.writeheader()

    for i, row in tqdm(enumerate(refined_i_note), desc='Write Refined I Note', ncols=75):
        writer.writerow(row)

        if not row['UnitNo'].isdigit():
            print(refined_c_note[i-1])
            print(row)




if __name__ == '__main__':
    main()