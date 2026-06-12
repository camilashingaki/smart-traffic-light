/*
  ESP32 — Controlador de LEDs do Semáforo Inteligente
  
  Recebe comandos via MQTT do Raspberry Pi e acende os LEDs corretos.
  
  Tópico MQTT: semaforo/fase
  Mensagens:
    "A"    → verde para carros, vermelho para pedestres
    "B"    → vermelho para carros, verde para pedestres
    "STOP" → vermelho para todos (segurança)
  
  Dependências (instale pelo Library Manager do Arduino IDE):
    - PubSubClient  (Nick O'Leary)
    - WiFi          (já incluída no ESP32)
  
  Como instalar:
    Arduino IDE → Tools → Manage Libraries → buscar "PubSubClient" → Install
*/

#include <WiFi.h>
#include <PubSubClient.h>

// ── Configurações de rede ─────────────────────────────────────────────────────
const char* WIFI_SSID     = "camila";      // hotspot do celular
const char* WIFI_PASSWORD = "camila72572";

// IP do Raspberry Pi na rede — descubra rodando "hostname -I" no Raspberry
const char* MQTT_BROKER = "test.mosquitto.org";   // <-- ALTERE PARA O IP DO RASPBERRY
const int   MQTT_PORT   = 1883;
const char* MQTT_TOPIC  = "semaforo/fase";
const char* CLIENT_ID   = "ESP32_Semaforo";

// ── Pinos dos LEDs ────────────────────────────────────────────────────────────
// Semáforo de carros
const int carroVerde     = 13;
const int carroAmarelo   = 14;
const int carroVermelho  = 12;

// Semáforo de pedestres 1 (lado leste)
const int pedestre1Verde    = 26;
const int pedestre1Vermelho = 25;

// Semáforo de pedestres 2 (lado oeste)
const int pedestre2Verde    = 18;
const int pedestre2Vermelho = 19;

// ── Duração do amarelo em ms (apenas visual, igual ao projeto) ────────────────
const int AMARELO_MS = 3000;   // 3 segundos

// ── Objetos WiFi e MQTT ───────────────────────────────────────────────────────
WiFiClient   espClient;
PubSubClient mqttClient(espClient);

// ── Estado atual ──────────────────────────────────────────────────────────────
String faseAtual = "";


// ── Funções de controle dos LEDs ──────────────────────────────────────────────

void todosVermelho() {
  // Coloca todos os semáforos em vermelho — usado em STOP e durante amarelo
  digitalWrite(carroVerde,      LOW);
  digitalWrite(carroAmarelo,    LOW);
  digitalWrite(carroVermelho,   HIGH);
  digitalWrite(pedestre1Verde,  LOW);
  digitalWrite(pedestre1Vermelho, HIGH);
  digitalWrite(pedestre2Verde,  LOW);
  digitalWrite(pedestre2Vermelho, HIGH);
}

void faseA() {
  /*
    Fase A: verde para carros, vermelho para pedestres
    Sequência: amarelo (3s) → verde carros
  */
  // Amarelo de transição (só se já havia uma fase anterior)
  if (faseAtual != "" && faseAtual != "A") {
    todosVermelho();
    digitalWrite(carroAmarelo, HIGH);
    delay(AMARELO_MS);
    digitalWrite(carroAmarelo, LOW);
  }

  // Verde para carros
  digitalWrite(carroVerde,    HIGH);
  digitalWrite(carroAmarelo,  LOW);
  digitalWrite(carroVermelho, LOW);

  // Vermelho para pedestres
  digitalWrite(pedestre1Verde,    LOW);
  digitalWrite(pedestre1Vermelho, HIGH);
  digitalWrite(pedestre2Verde,    LOW);
  digitalWrite(pedestre2Vermelho, HIGH);

  Serial.println("[FASE A] Verde carros | Vermelho pedestres");
}

void faseB() {
  /*
    Fase B: vermelho para carros, verde para pedestres
    Sequência: amarelo (3s) → verde pedestres
  */
  // Amarelo de transição
  if (faseAtual != "" && faseAtual != "B") {
    todosVermelho();
    digitalWrite(carroAmarelo, HIGH);
    delay(AMARELO_MS);
    digitalWrite(carroAmarelo, LOW);
  }

  // Vermelho para carros
  digitalWrite(carroVerde,    LOW);
  digitalWrite(carroAmarelo,  LOW);
  digitalWrite(carroVermelho, HIGH);

  // Verde para pedestres
  digitalWrite(pedestre1Verde,    HIGH);
  digitalWrite(pedestre1Vermelho, LOW);
  digitalWrite(pedestre2Verde,    HIGH);
  digitalWrite(pedestre2Vermelho, LOW);

  Serial.println("[FASE B] Vermelho carros | Verde pedestres");
}


// ── Callback MQTT — chamado quando chega mensagem ─────────────────────────────

void onMensagem(char* topic, byte* payload, unsigned int length) {
  // Converte payload para String
  String mensagem = "";
  for (unsigned int i = 0; i < length; i++) {
    mensagem += (char)payload[i];
  }
  mensagem.trim();

  Serial.print("MQTT recebido [");
  Serial.print(topic);
  Serial.print("]: ");
  Serial.println(mensagem);

  if (mensagem == "A") {
    faseA();
    faseAtual = "A";
  } else if (mensagem == "B") {
    faseB();
    faseAtual = "B";
  } else if (mensagem == "STOP") {
    todosVermelho();
    faseAtual = "STOP";
    Serial.println("[STOP] Todos vermelho — sistema encerrado");
  } else {
    Serial.print("Mensagem desconhecida: ");
    Serial.println(mensagem);
  }
}


// ── Conexão Wi-Fi ─────────────────────────────────────────────────────────────

void conectarWiFi() {
  Serial.print("Conectando ao Wi-Fi: ");
  Serial.println(WIFI_SSID);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int tentativas = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    tentativas++;
    if (tentativas > 30) {
      Serial.println("\nFalha ao conectar. Reiniciando...");
      ESP.restart();
    }
  }

  Serial.println("\nWi-Fi conectado!");
  Serial.print("IP do ESP32: ");
  Serial.println(WiFi.localIP());
}


// ── Conexão MQTT ──────────────────────────────────────────────────────────────

void conectarMQTT() {
  while (!mqttClient.connected()) {
    Serial.print("Conectando ao broker MQTT...");

    if (mqttClient.connect(CLIENT_ID)) {
      Serial.println(" conectado!");

      // Inscreve no tópico do semáforo
      mqttClient.subscribe(MQTT_TOPIC);
      Serial.print("Inscrito no tópico: ");
      Serial.println(MQTT_TOPIC);

    } else {
      Serial.print(" falhou (rc=");
      Serial.print(mqttClient.state());
      Serial.println("). Tentando novamente em 3s...");
      delay(3000);
    }
  }
}


// ── Setup ─────────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== Semáforo Inteligente — ESP32 ===");

  // Configura pinos como saída
  pinMode(carroVerde,     OUTPUT);
  pinMode(carroAmarelo,   OUTPUT);
  pinMode(carroVermelho,  OUTPUT);
  pinMode(pedestre1Verde,    OUTPUT);
  pinMode(pedestre1Vermelho, OUTPUT);
  pinMode(pedestre2Verde,    OUTPUT);
  pinMode(pedestre2Vermelho, OUTPUT);

  // Estado inicial: todos vermelho (segurança)
  todosVermelho();
  Serial.println("LEDs inicializados — todos vermelho.");

  // Conecta Wi-Fi e MQTT
  conectarWiFi();
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  mqttClient.setCallback(onMensagem);
  conectarMQTT();

  Serial.println("Sistema pronto. Aguardando comandos...");
}


// ── Loop ──────────────────────────────────────────────────────────────────────

void loop() {
  // Reconecta automaticamente se cair
  if (!mqttClient.connected()) {
    Serial.println("MQTT desconectado. Reconectando...");
    conectarMQTT();
  }

  // Processa mensagens MQTT recebidas
  mqttClient.loop();
}
