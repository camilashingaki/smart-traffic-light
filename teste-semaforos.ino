int carroVerde = 13;
int carroAmarelo = 14;
int carroVermelho = 12;

int pedestre1Verde = 26;
int pedestre1Vermelho = 25;

int pedestre2Verde = 35;
int pedestre2Vermelho = 34;

int tempoFase1 = 5000; // Carros passam
int tempoFase2 = 2000; // Alerta
int tempoFase3 = 5000; // Pedestres atravessam

void setup() {
  pinMode(carroVerde, OUTPUT);
  pinMode(carroAmarelo, OUTPUT);
  pinMode(carroVermelho, OUTPUT);
  pinMode(pedestre1Verde, OUTPUT);
  pinMode(pedestre1Vermelho, OUTPUT);
  pinMode(pedestre2Verde, OUTPUT);
  pinMode(pedestre2Vermelho, OUTPUT);
}

void fase1() {
  digitalWrite(carroVerde, HIGH);
  digitalWrite(carroAmarelo, LOW);
  digitalWrite(carroVermelho, LOW);
  digitalWrite(pedestre1Verde, LOW);
  digitalWrite(pedestre1Vermelho, HIGH);
  digitalWrite(pedestre2Verde, HIGH);
  digitalWrite(pedestre2Vermelho, LOW);
  delay(tempoFase1);
}

void fase2() {
  digitalWrite(carroVerde, LOW);
  digitalWrite(carroAmarelo, HIGH);
  digitalWrite(carroVermelho, LOW);
  digitalWrite(pedestre1Verde, LOW);
  digitalWrite(pedestre1Vermelho, HIGH);
  digitalWrite(pedestre2Verde, HIGH);
  digitalWrite(pedestre2Vermelho, LOW);
  delay(tempoFase2);
}

void fase3() {
  digitalWrite(carroVerde, LOW);
  digitalWrite(carroAmarelo, LOW);
  digitalWrite(carroVermelho, HIGH);
  digitalWrite(pedestre1Verde, HIGH);
  digitalWrite(pedestre1Vermelho, LOW);
  digitalWrite(pedestre2Verde, LOW);
  digitalWrite(pedestre2Vermelho, HIGH);
  delay(tempoFase3);
}

void loop() {
  fase1();
  fase2();
  fase3();
}