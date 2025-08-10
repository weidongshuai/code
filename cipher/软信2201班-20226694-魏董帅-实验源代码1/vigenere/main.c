#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// ʹ����Կ���ı����м���
char* vigenere_encrypt(char* text, char* key) {
    int textLen = strlen(text);
    int keyLen = strlen(key);
    char* encrypted = (char*)malloc(textLen + 1);
    int i;
    for (i = 0; i < textLen; i++) {
        // ��ÿ���ַ����м���
        encrypted[i] = ((text[i] - 'A' + key[i % keyLen] - 'A') % 26) + 'A';
    }
    encrypted[i] = '\0';
    return encrypted;
}

// ʹ����Կ���ı����н���
char* vigenere_decrypt(char* text, char* key) {
    int textLen = strlen(text);
    int keyLen = strlen(key);
    char* decrypted = (char*)malloc(textLen + 1);
    int i;
    for (i = 0; i < textLen; i++) {
        // ��ÿ���ַ����н���
        decrypted[i] = ((text[i] - 'A' - (key[i % keyLen] - 'A') + 26) % 26) + 'A';
    }
    decrypted[i] = '\0';
    return decrypted;
}

int main() {
    char text[100]; // ����
    char key[100]; // ��Կ

    printf("������Ҫ���ܵ��ı�(��д): ");
    scanf("%s", text);

    printf("��������Կ: ");
    scanf("%s", key);

    // �����ı�
    char* encryptedText = vigenere_encrypt(text, key);
    printf("���ܺ���ı�: %s\n", encryptedText);

    // �����ı�
    char* decryptedText = vigenere_decrypt(encryptedText, key);
    printf("���ܺ���ı�: %s\n", decryptedText);

    free(encryptedText);
    free(decryptedText);

    return 0;
}
