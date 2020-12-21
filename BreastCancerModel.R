# Breast Cancer Prediction

data = read.csv('Data/wdbc.data', header=FALSE)

summary(data)
dim(data) #569 x 32
# Add in column names from the documentation
colnames(data) = c('ID', 'Diag', 'radius', 'texture','perimeter', 'area', 'smoothness', 'compactness', 'concavity', 
                   'concave points', 'symmetry', 'fractal dim', 'se_radius', 'se_texture','se_perim', 'se_area', 'se_smooth', 
                   'se_compact', 'se_concavity', 'se_concpts', 'se_symm', 'se_fractdim', 'max_rad', 'max_text', 'max_perim', 
                   'max_area', 'max_smooth', 'max_compact', 'max_concavity', 'max_concpts', 'max_symm', 'max_fracdim')
pairs(data[,2:12])
pairs(data[,c(2,13:22)])
pairs(data[,c(2,23:32)])
