# extract composition and occurrence.
rm(list = ls())

g_data = read.csv('C:\\Users\\idehz\\Desktop\\Subcellular_data/g_data.csv', header = F)
alphabet = c('A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y')

# head(g_data,1)
# g_data[1,4]

occur = matrix(nrow=length(g_data[,1]),ncol = length(alphabet),0)
comp = matrix(nrow=length(g_data[,1]),ncol = length(alphabet),0)
bigram0 = matrix(nrow=length(g_data[,1]),ncol = length(alphabet)*length(alphabet),0)
bigram1 = matrix(nrow=length(g_data[,1]),ncol = length(alphabet)*length(alphabet),0)
bigram2 = matrix(nrow=length(g_data[,1]),ncol = length(alphabet)*length(alphabet),0)
#occur1 = matrix(nrow=length(g_data[,1]),ncol = length(alphabet),0)
#comp1 = matrix(nrow=length(g_data[,1]),ncol = length(alphabet),0)
#rm(occur1,comp1)
# Occurrence & Composition old - revised is after:

# for (i in 1:length(g_data[,1])) 
# {
#   cont = 0
#   for(k in 1:length(alphabet))
#   {
#     cont = 0
#     for (j in 1:nchar(as.character(g_data[i,4])))
#     {
#       s = strsplit(as.character(g_data[i,4]),split = '')
#       if(as.character(alphabet[k]) == as.character(s[[1]][j]))
#       {
#         cont = cont + 1
#       }
#     }
#     occur[i,k]=cont
#     comp[i,k]=round(cont/nchar(as.character(g_data[i,4])),2)
#   }
# }

# occur and composition revised:

for (i in 1:length(g_data[,1])) 
{
  for (j in 1:nchar(as.character(g_data[i,4])))
  {
    s = strsplit(as.character(g_data[i,4]),split = '')
    for(k in 1:length(alphabet))
    {
      if(as.character(alphabet[k]) == as.character(s[[1]][j]))
      {
        occur[i,k] = occur[i,k]+1
      }
    }
  }
  for(k in 1:length(alphabet))
  {
    comp[i,k]=round(occur[i,k]/nchar(as.character(g_data[i,4])),2)
  }
}

occur[1:5,1:5]
comp[1:5,1:5]
#sum(bigram0[1,])
sum(occur[1,])
sum(comp[1,])
nchar(as.character(g_data[1,4]))
sum(occur[2,])
sum(comp[2,])
nchar(as.character(g_data[2,4]))
sum(occur[200,])
sum(comp[200,])
nchar(as.character(g_data[200,4]))
sum(occur[400,])
sum(comp[400,])
nchar(as.character(g_data[400,4]))
sum(occur[523,])
sum(comp[523,])
nchar(as.character(g_data[523,4]))
# Bigram with 0 distance:

for (i in 1:length(g_data[,1])) 
{
  for (j in 1:(nchar(as.character(g_data[i,4]))-1))
  {
    s = strsplit(as.character(g_data[i,4]),split = '')
    for(k in 1:length(alphabet))
    {
      if(as.character(alphabet[k]) == as.character(s[[1]][j]))
      {
        for(l in 1:length(alphabet))
        {
          if(as.character(alphabet[l]) == as.character(s[[1]][j+1]))
          {
            bigram0[i,k*l]= bigram0[i,k*l]+1
          }
        }
      }
    }
  }
}
bigram0[1:5,1:5]
sum(bigram0[1,])
nchar(as.character(g_data[1,4]))-1
sum(bigram0[2,])
nchar(as.character(g_data[2,4]))-1
sum(bigram0[200,])
nchar(as.character(g_data[200,4]))-1
sum(bigram0[523,])
nchar(as.character(g_data[523,4]))-1
sum(bigram0[400,])
nchar(as.character(g_data[400,4]))-1

# Bigram with 1 distance:

for (i in 1:length(g_data[,1])) 
{
  for (j in 1:(nchar(as.character(g_data[i,4]))-2))
  {
    s = strsplit(as.character(g_data[i,4]),split = '')
    for(k in 1:length(alphabet))
    {
      if(as.character(alphabet[k]) == as.character(s[[1]][j]))
      {
        for(l in 1:length(alphabet))
        {
          if(as.character(alphabet[l]) == as.character(s[[1]][j+2]))
          {
            bigram1[i,k*l]= bigram1[i,k*l]+1
          }
        }
      }
    }
  }
}
bigram1[1:5,1:5]
sum(bigram1[1,])
nchar(as.character(g_data[1,4]))-2
sum(bigram1[2,])
nchar(as.character(g_data[2,4]))-2
sum(bigram1[200,])
nchar(as.character(g_data[200,4]))-2
sum(bigram1[523,])
nchar(as.character(g_data[523,4]))-2
sum(bigram1[400,])
nchar(as.character(g_data[400,4]))-2

############ Bigram with 2 distance:

for (i in 1:length(g_data[,1])) 
{
  for (j in 1:(nchar(as.character(g_data[i,4]))-3))
  {
    s = strsplit(as.character(g_data[i,4]),split = '')
    for(k in 1:length(alphabet))
    {
      if(as.character(alphabet[k]) == as.character(s[[1]][j]))
      {
        for(l in 1:length(alphabet))
        {
          if(as.character(alphabet[l]) == as.character(s[[1]][j+3]))
          {
            bigram2[i,k*l]= bigram2[i,k*l]+1
          }
        }
      }
    }
  }
}
bigram2[1:5,1:5]
sum(bigram2[1,])
nchar(as.character(g_data[1,4]))-3
sum(bigram2[2,])
nchar(as.character(g_data[2,4]))-3
sum(bigram2[200,])
nchar(as.character(g_data[200,4]))-3
sum(bigram2[523,])
nchar(as.character(g_data[523,4]))-3
sum(bigram2[400,])
nchar(as.character(g_data[400,4]))-3

save(occur,comp,file = 'C:\\Users\\idehz\\Desktop\\Subcellular_data/gram_p_features.rdata')
save(bigram0,bigram1,bigram2,file = 'C:\\Users\\idehz\\Desktop\\Subcellular_data/bigram_p_features.rdata')  
write.csv(occur,file = 'C:\\Users\\idehz\\Desktop\\Subcellular_data/occur.csv')
write.csv(comp,file = 'C:\\Users\\idehz\\Desktop\\Subcellular_data/comp.csv')
write.csv(bigram0,file = 'C:\\Users\\idehz\\Desktop\\Subcellular_data/bigram0.csv')
write.csv(bigram1,file = 'C:\\Users\\idehz\\Desktop\\Subcellular_data/bigram1.csv')
write.csv(bigram2,file = 'C:\\Users\\idehz\\Desktop\\Subcellular_data/bigram2.csv')
#dim(occur)
######################
## Test
#######################

# s[1]
# s[2]
# cont  =0 
# for (j in 1:nchar(as.character(g_data[1,4])))
# {
#   s = strsplit(as.character(g_data[1,4]),split = '')
#   if(as.character(alphabet[1]) == as.character(s[[1]][j]))
#   {
#     cont = cont + 1
#   }
# }
# print(cont)
