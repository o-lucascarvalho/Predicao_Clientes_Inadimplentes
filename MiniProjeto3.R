#Mini Projeto 3 - Previsão de Inadimplência com Machine Learning e Power BI

#Definição da pasta de trabalho
setwd("D:/DSA-PowerBI/Cap15")
getwd()

#instalação de pacotes
install.packages("Amelia")
install.packages("caret")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("reshape")
install.packages("randomForest")
install.packages("e1071")

#Carregamento dos pacotes
library(Amelia)
library(ggplot2)
library(caret)
library(reshape)
library(randomForest)
library(dplyr)
library(e1071)

#Carregamento do dataset
dados_clientes <- read.csv("dados/dataset.csv")

#Visualização de dados e estrutura
View(dados_clientes)
dim(dados_clientes)
str(dados_clientes)
summary(dados_clientes)

#             Análise Exploratório, Limpeza e Transformação dos dados

#Remoção da primeira coluna ID
dados_clientes$ID <- NULL
dim(dados_clientes)
View(dados_clientes)

#Renomear a coluna de Classe
colnames(dados_clientes)
colnames(dados_clientes)[24] <- "Inadimplente"
colnames(dados_clientes)
View(dados_clientes)

#Verificando valores ausentes e removendo-os
sapply(dados_clientes, function(x) sum(is.na(x)))
?missmap
missmap(dados_clientes, main ="Valores Missing Observados")
dados_clientes <- na.omit(dados_clientes)

#Conversão de atributos genero, escolaridade, estado civil e idade para fatores (Categorias)

#Renomear colunas categóricas
colnames(dados_clientes)
colnames(dados_clientes)[2] <- "Genero"
colnames(dados_clientes)[3] <- "Escolaridade"
colnames(dados_clientes)[4] <- "Estado_civil"
colnames(dados_clientes)[5] <- "Idade"
colnames(dados_clientes)
View(dados_clientes)

#Conversão Genero
View(dados_clientes$Genero)
str(dados_clientes$Genero)
summary(dados_clientes$Genero)
?cut
dados_clientes$Genero <- cut(dados_clientes$Genero,
                             c(0,1,2),
                             labels = c("Masculino",
                                        "Feminino"))
View(dados_clientes$Genero)
str(dados_clientes$genero)

#Conversão Escolaridade
str(dados_clientes$Escolaridade)
summary(dados_clientes$Escolaridade)
dados_clientes$Escolaridade <- cut(dados_clientes$Escolaridade,
                                   c(0,1,2,3,4),
                                   labels = c("Pos Graduado",
                                              "Graduado",
                                              "Ensino Medio",
                                              "Outros"))
View(dados_clientes$Escolaridade)
str(dados_clientes$Escolaridade)
summary(dados_clientes$Escolaridade)

#Conversão Estado Civil
str(dados_clientes$Estado_civil)
summary(dados_clientes$Estado_civil)
dados_clientes$Estado_civil <- cut(dados_clientes$Estado_civil,
                                   c(-1,0,1,2,3),
                                   labels = c("Desconhecido",
                                              "Casado",
                                              "Solteiro",
                                              "outro"))
View(dados_clientes$Estado_civil)
str(dados_clientes$Estado_civil)
summary(dados_clientes$Estado_civil)

#Conversao idade com faixa etária
str(dados_clientes$Idade)
summary(dados_clientes$Idade)
hist(dados_clientes$Idade)
dados_clientes$Idade <- cut(dados_clientes$Idade,
                            c(0,30,50,100),
                            labels = c("Jovem",
                                       "Adulto",
                                       "Idoso"))
View(dados_clientes$Idade)
str(dados_clientes$Idade)
summary(dados_clientes$Idade)
View(dados_clientes)


#Conversão da variável para o tipo fator
dados_clientes$PAY_0 <- as.factor(dados_clientes$PAY_0)
dados_clientes$PAY_2 <- as.factor(dados_clientes$PAY_2)
dados_clientes$PAY_3 <- as.factor(dados_clientes$PAY_3)
dados_clientes$PAY_4 <- as.factor(dados_clientes$PAY_4)
dados_clientes$PAY_5 <- as.factor(dados_clientes$PAY_5)
dados_clientes$PAY_6 <- as.factor(dados_clientes$PAY_6)

#Visualização de dataset após conversão
str(dados_clientes)
sapply(dados_clientes, function(x) sum(is.na(x)))
missmap(dados_clientes, main ="Valores Missing Observados")
dados_clientes <- na.omit(dados_clientes)
missmap(dados_clientes, main = "Valores Missing Observados")
dim(dados_clientes)

#Convertendo variavel inadimplente para fator
str(dados_clientes$Inadimplente)
colnames(dados_clientes)
dados_clientes$Inadimplente <- as.factor(dados_clientes$Inadimplente)
str(dados_clientes$Inadimplente)
View(dados_clientes)

#Total de Inadimplesntes X Não inadimplentes
table(dados_clientes$Inadimplente)

#Porcentagem entre as duas classes
prop.table(table(dados_clientes$Inadimplente))

#plot da distribuição
qplot(Inadimplente, data = dados_clientes, geom ="bar") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

set.seed(12345)

#Amostragem Estratificada
#Seleciona as linhas de acordo com a variável inadimplente como strata
?createDataPartition
indice <- createDataPartition(dados_clientes$Inadimplente, p = 0.75, list = FALSE)
dim(indice)

#Definição de dados de treinamento como subconjunto do conjunto de dados Original
#com números de indice de linha e todas as colunas
dados_treino <- dados_clientes[indice,]
dim(dados_treino)
table(dados_treino$Inadimplente)

#Porcentagem entre as classes
prop.table(table(dados_treino$Inadimplente))

#numero de registro de treino
dim(dados_treino)

#Comparação das porcentagens entre as classes de treino e dados original
compara_dados <- cbind(prop.table(table(dados_treino$Inadimplente)),
                      prop.table(table(dados_clientes$Inadimplente)))
colnames(compara_dados) <- c("Treinamento", "Original")
compara_dados

#Melt Data - Converte colunas em linhas
?reshape2::melt
melt_compara_dados <- melt(compara_dados)
melt_compara_dados

#plot para ver a distribuição do trienamento x original
ggplot(melt_compara_dados, aes(x = X1, y = value)) +
  geom_bar( aes(fill = X2), stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#Valores que não estão no dataset de treino, irão para o dataset de teste
dados_teste <- dados_clientes[-indice,]
dim(dados_teste)
dim(dados_treino)


####Criação do Modelo de Machine Learning ########


?randomForest
modelo_v1 <- randomForest(Inadimplente ~ ., data = dados_treino)
modelo_v1

#Avaliação do modelo
plot(modelo_v1)

#Previsoes com dados de teste
previsoes_v1 <- predict(modelo_v1, dados_teste)

#Confusion Matrix
?caret::confusionMatrix
cm_v1 <- caret::confusionMatrix(previsoes_v1, dados_teste$Inadimplente, positive = "1")
cm_v1

#calculando Precision, Recall e F1-score, métricas de avaliação do modelo preditivo
y <- dados_teste$Inadimplente
y_pred_v1 <- previsoes_v1

precision <- posPredValue(y_pred_v1, y)
precision

recall <- sensitivity(y_pred_v1, y)
recall

f1 <- (2 * precision * recall) / (precision + recall)
f1

#Balanceamento de classe
install.packages("DMwR")
library(DMwr)
?SMOTE

#Aplicação de SMOTE
table(dados_treino$Inadimplente)
prop.table(table(dados_treino$Inadimplente))
set.seed(9560)
dados_treino_bal <- SMOTE(Inadimplente ~ ., data = dados_treino)
table(dados_treino_bal$Inadimplente)
prop.table(table(dados_treino_bal$Inadimplente))

#Construção da segunda versão do modelo.
modelo_v2 <- randomForest(Inadimplente ~ ., data = dados_treino_bal)
modelo_v2

#Avaliação do modelo
plot(modelo_v2)

#Previsão com dados de teste
previsoes_v2 <- predict(modelo_v2, dados_teste)

#Confusion Matrix
?caret::confusionMatrix
cm_v2 <- caret::confusionMatrix(previsoes_v2, dados_teste$Inadimplente, positive = "1")
cm_v2

#Calculando Precision, Recall e F1-score, métricas de avaliação do modelo preditivo
y <- dados_teste$Inadimplente
y_pred_v2 <- previsoes_v2

precision <- posPredValue(y_pred_v2, y)
precision

recall <- sensitivity(y_pred_v2, y)
recall

f1 <- (2 * precision * recall) / (precision + recall)
f1

#Importancia das variáveis preditoras para as previsoes
View(dados_treino_bal)
varImpPlot(modelo_v2)

#Obter as variaveis mais importantes
imp_var <- importance(modelo_v2)
varImportance <- data.frame(Variables = row.names(imp_var),
                            Importance = round(imp_var[ ,'MeanDecreaseGini'],2))

#Criando o rank de variaveis baseado na importancia
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#', dense_rank(desc(Importance))))


#Usando ggplot2 para visualizar a importancia relativa das variaveis
ggplot(rankImportance,
       aes(x = reorder(Variables, Importance),
           y = Importance,
           fill = Importance)) +
  geom_bar(stat='identity') +
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust = 0,
            vjust = 0.55,
            size = 4,
            colour = 'red') +
  labs(x = 'variables') + 
  coord_flip()




#Terceira versão do modelo
colnames(dados_treino_bal)
?randomForest
modelo_v3 <- randomForest(Inadimplente ~ PAY_0 + PAY_2 + PAY_3 + PAY_AMT1 + PAY_AMT2 + PAY_5 + BILL_AMT1, data = dados_treino_bal)
modelo_v3

#Avaliação do modelo
plot(modelo_v3)

#Previsoes com testes
previsoes_v3 <- predict(modelo_v3, dados_teste)

#Confusion Matrix
?caret::confusionMatrix
cm_v3 <- caret::confusionMatrix(previsoes_v3, dados_teste$Inadimplente, positive = "1")
cm_v3

#Calculo de precision, recall e f1-score, métricas de avaliação do modelo preditivo
y <- dados_teste$Inadimplente
y_pred_v3 <- previsoes_v3

precision <- posPredValue(y_pred_v3, y)
precision

recall <- sensitivity(y_pred_v3, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

#Salvando o modelo em disco
saveRDS(modelo_v3, file = "modelo/modelo_v3.rds")

#Carregando o modelo
modelo_final <- readRDS("modelo/modelo_v3.rds")

#Previsão com novo cliente
PAY_0 <- c(0, 0, 0)
PAY_2 <- c(0, 0, 0)
PAY_3 <- c(1, 0, 0)
PAY_AMT1 <- c(1100, 1000, 1200)
PAY_AMT2 <- c(1500, 1300, 1150)
PAY_5 <- c(0, 0, 0)
BILL_AMT1 <- c(350, 420, 280)

#Concatenar em um data frame
novos_clientes <- data.frame(PAY_0, PAY_2, PAY_3, PAY_AMT1, PAY_AMT2, PAY_5, BILL_AMT1)
View(novos_clientes)

str(novos_clientes)

#Conversao dos dados de novos clientes

novos_clientes$PAY_0 <- factor(novos_clientes$PAY_0, levels = levels(dados_treino_bal$PAY_0))
novos_clientes$PAY_2 <- factor(novos_clientes$PAY_2, levels = levels(dados_treino_bal$PAY_2))
novos_clientes$PAY_3 <- factor(novos_clientes$PAY_3, levels = levels(dados_treino_bal$PAY_3))
novos_clientes$PAY_5 <- factor(novos_clientes$PAY_5, levels = levels(dados_treino_bal$PAY_5))
str(novos_clientes)

#Previsoes
previsoes_novo_cliente <- predict(modelo_final, novos_clientes)
View(previsoes_novo_cliente)












































































































