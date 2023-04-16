class ACDCDataset(Dataset):
    def __init__(self, root_dir, mode='train', transforms=None, vxm=False):
        self.transforms = transforms
        self.is_test = mode == 'test'
        self.vxm = vxm
        self.class_names = ['background', 'rv', 'my', 'lv']
        self.num_classes = len(self.class_names)

        split_dir = osp.join(root_dir, 'testing' if self.is_test else 'training')
        self.patients_list_dir = [osp.join(split_dir, p) for p in natsorted(os.listdir(split_dir))]

    def __len__(self):
        return len(self.patients_list_dir)

    def __getitem__(self, idx):
        patient_dir = self.patients_list_dir[idx]
        info_cfg = self.read_configfile(osp.join(patient_dir, 'Info.cfg'))
        ed = info_cfg.getint('ED')
        es = info_cfg.getint('ES')
        patient_name = osp.basename(patient_dir)

        # End diastolic data
        nii_img_ed = nib.load(osp.join(patient_dir, '{}_frame{:02d}.nii.gz'.format(patient_name, ed)))
        nii_mask_ed = nib.load(osp.join(patient_dir, '{}_frame{:02d}_gt.nii.gz'.format(patient_name, ed)))
        img_ed = np.swapaxes(nii_img_ed.get_fdata(), 0, 2)
        mask_ed = np.swapaxes(nii_mask_ed.get_fdata(), 0, 2)

        # End systolic data
        nii_img_es = nib.load(osp.join(patient_dir, '{}_frame{:02d}.nii.gz'.format(patient_name, es)))
        nii_mask_es = nib.load(osp.join(patient_dir, '{}_frame{:02d}_gt.nii.gz'.format(patient_name, es)))
        img_es = np.swapaxes(nii_img_es.get_fdata(), 0, 2)
        mask_es = np.swapaxes(nii_mask_es.get_fdata(), 0, 2)

        imgs = np.stack((img_es, img_ed), axis=3)
        masks = np.stack((mask_es, mask_ed), axis=3)

        img_meta = dict(nii_img_ed.header)
        img_meta['affine'] = nii_img_ed.affine
        mask_meta = dict(nii_mask_ed.header)
        mask_meta['affine'] = nii_mask_ed.affine

        data = {'img': imgs,
                'mask': masks,
                'img_meta': img_meta,
                'mask_meta': mask_meta,
                'patient': patient_name, 'ed': ed, 'es': es}
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def read_configfile(self, file_dir):
        with open(file_dir, 'r') as f:
            config_string = '[dummy_section]\n' + f.read()
        config = configparser.ConfigParser()
        config.read_string(config_string)
        return config['dummy_section']

    @staticmethod
    def collate_fn(batch):
        ret = UserDict(**default_collate(batch))
        return ret

