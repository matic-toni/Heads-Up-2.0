import os
import shutil

class DirectoryOrganizer():

    def __init__(self):
        pass

    def divide_samples(self, srcdir, dstdir, by='labels', num_categories=2):
        '''
        Divides samples from srcdir based on the criteria defined by the 'by' argument, it can be either
        'labels' or 'processed'. Default value is 'labels'. Method expects filenames from the source directory
        to respect following format: <multiple characters>_<is processed>_<target class>.ext. Both <is processed> and
        <target class> need to be 1 character long so that parsing process ends successfully. Based on the values from
        those two fields sufficient directories with names will be created as subdirectories in the destination directory.
        '''
        destination_path = os.path.join('.', dstdir)
        os.mkdir(destination_path)
        print(f"Directory '{destination_path}' created")

        samples_by_category = [0] * num_categories
        categories = set()
        for entry in os.listdir(srcdir):
            if by == 'labels':
                category = entry.split('_')[2].split('.')[0]
            else:
                category = entry.split('_')[1]
            
            if category not in categories:
                categories.add(category)
                os.mkdir(os.path.join(destination_path, category))
            
            samples_by_category[int(category)] += 1

            src = os.path.abspath(os.path.join('.', srcdir, entry))
            dst = os.path.abspath(os.path.join('.', dstdir, category, entry))
            shutil.copyfile(src, dst)

        total = sum(samples_by_category)
        for category, count in enumerate(samples_by_category):
            print(f'Category {category}: {format(100*count/total, ".2f")}%')
        print('Total: ', total)