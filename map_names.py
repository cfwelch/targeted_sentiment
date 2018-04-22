
import NLU
import dicts
import random


def main():
	kez = dicts.getEECSdict().keys();
	to_map = ['492', '554', '574', '402', '270', '595', '597', '699', '525', '330', '381', '480', '560', '463', '493', '531', '423', '568', '482', '481', '592', '553', '183', '414', '455', '564', '496', '500', '413', '571', '551', '522', '518', '558', '320', '312', '406', '442', '509', '600', '215', '427', '540', '517', '555', '477', '411', '489', '419', '470', '543', '499', '501', '594', '475', '376', '417', '767', '570', '582', '445', '441', '280', '458', '587', '995', '528', '373', '575', '573', '521', '566', '671', '460', '545', '598', '216', '494', '502', '418', '519', '695', '452', '599', '584', '151', '388', '565', '586', '203', '542', '451', '692', '285', '250', '429', '461', '556', '578', '497', '628', '523', '530', '520', '510', '541', '467', '485', '487', '511', '443', '567', '281', '569', '438', '561', '398', '311', '498', '550', '473', '351', '421', '588', '589', '484', '334', '425', '483', '434', '101', '399', '301', '579', '583', '562', '453', '230', '755', '591', '314', '430', '370', '282', '478', '401'];
	f_ile = open("EECS_annotated_samples");
	lines = f_ile.readlines();
	f_ile.close();
	f_out = open("out_", "w");
	for line in lines:
		t_line = line;
		for KZ in kez:
			if KZ in t_line:
				t_line = t_line.replace(KZ, to_map[kez.index(KZ)]);
		f_out.write(t_line);

	f_out.flush();
	f_out.close();



if __name__ == "__main__":
	main()
