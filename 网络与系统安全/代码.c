#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/ether.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include <netinet/ip_icmp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <linux/if_packet.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <time.h>
#include <termios.h>
#include <fcntl.h>
#include <errno.h>
#include <syslog.h>

// ����ͳ�����ݰ������Ϣ�Ľṹ��
typedef struct PacketStats {
    char startTime[30];
    char endTime[30];
    int macBroadcastCount;
    int macShortCount;
    int macLongCount;
    int macByteCount;
    int macPacketCount;
    int bitPerSecond;
    int macByteSpeed;
    int macPacketSpeed;
    int ipBroadcastCount;
    int ipByteCount;
    int ipPacketCount;
    int udpPacketCount;
    int icmpPacketCount;
    int icmpRedirectCount;
    int icmpDestinationCount;
} PacketStats;

// �������ڲ������ݰ���ԭʼ�׽���
int create_socket() {
    int sockfd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if (sockfd == -1) {
        syslog(LOG_ERR, "Socket creation failed: %s", strerror(errno));
        exit(EXIT_FAILURE);
    }
    return sockfd;
}

// �����׽��ֽ��ջ�������С(��ѡ,�ɸ���ʵ���������)
void set_socket_buffer_size(int sockfd, int buffer_size) {
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &buffer_size, sizeof(buffer_size)) == -1) {
        syslog(LOG_ERR, "Failed to set socket buffer size: %s", strerror(errno));
    }
}

// ���׽��ְ󶨵�ָ��������ӿ�(�������ӿ�����֪)
int bind_to_interface(int sockfd, const char *interface_name) {
    struct sockaddr_ll socket_address;
    memset(&socket_address, 0, sizeof(socket_address));
    socket_address.sll_family = AF_PACKET;

    // ��ȡ�ӿ�����
    struct ifreq ifr;
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, interface_name, IFNAMSIZ - 1);
    if (ioctl(sockfd, SIOCGIFINDEX, &ifr) == -1) {
        syslog(LOG_ERR, "Failed to get interface index: %s", strerror(errno));
        return -1;
    }

    socket_address.sll_ifindex = ifr.ifr_ifindex;

    // ���׽��ֵ��ӿ�
    if (bind(sockfd, (struct sockaddr *)&socket_address, sizeof(socket_address)) == -1) {
    syslog(LOG_ERR, "Failed to bind socket to interface: %s", strerror(errno));
        return -1;
    }

    return 0;
}

// �ײ�ģ��������,��ʼ����׼�������ڲ������ݰ����׽��ֵ���Դ
int init_packet_capture(const char *interface_name, int buffer_size) {
    int sockfd = create_socket();
    if (sockfd == -1) {
        return -1;
    }

    // ���ý��ջ�������С(�ɸ���ʵ�ʵ������ʵ�ֵ)
    set_socket_buffer_size(sockfd, buffer_size);

    // �󶨵�ָ���ӿ�
    if (bind_to_interface(sockfd, interface_name) == -1) {
        close(sockfd);
        return -1;
    }

    return sockfd;
}

// �� MAC ��ַ�Ӷ�������ʽת��Ϊ�ַ�����ʽ(��������)
char *mac_ntoa(const unsigned char *mac) {
    static char mac_str[18];
    sprintf(mac_str, "%.2x:%.2x:%.2x:%.2x:%.2x:%.2x",
            mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    return mac_str;
}

// �� IP ��ַ�Ӷ�������ʽת��Ϊ�ַ�����ʽ(��������)
char *ip_ntoa(const struct in_addr *addr) {
    static char ip_str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, addr, ip_str, INET_ADDRSTRLEN);
    return ip_str;
}

// �в�ģ��:MAC �㴦����غ���ʾ��
void print_ethernet(const struct ether_header *eth) {
    printf("Ethernet Header:\n");
    printf("Source MAC: %s\n", mac_ntoa(eth->ether_shost));
    printf("Destination MAC: %s\n", mac_ntoa(eth->ether_dhost));
    printf("Ethernet Type: 0x%04x\n", ntohs(eth->ether_type));
}

// �в�ģ��:IP �㴦����غ���ʾ��
void print_ip(const struct iphdr *ip) {
    printf("IP Header:\n");
    char *src_ip = ip_ntoa((struct in_addr *)&ip->saddr);
    char *dst_ip = ip_ntoa((struct in_addr *)&ip->daddr);
    printf("Version: %d\n", ip->version);
    printf("Header Length: %d bytes\n", ip->ihl * 4);
    printf("Type of Service: 0x%02x\n", ip->tos);
    printf("Total Length: %d bytes\n", ntohs(ip->tot_len));
    printf("Identification: 0x%04x\n", ntohs(ip->id));
    printf("Flags: 0x%02x\n", ip->frag_off >> 13);
    printf("Fragment Offset: %d\n", (ip->frag_off & 0x1fff) * 8);
    printf("Time to Live: %d\n", ip->ttl);
    printf("Protocol: %d\n", ip->protocol);
    printf("Header Checksum: 0x%04x\n", ntohs(ip->check));
    printf("Source IP: %s\n", src_ip);
    printf("Destination IP: %s\n", dst_ip);
}

// �в�ģ��:TCP ������غ���ʾ��
void print_tcp(const struct tcphdr *tcp) {
    printf("TCP Header:\n");
    printf("Source Port: %d\n", ntohs(tcp->source));
    printf("Destination Port: %d\n", ntohs(tcp->dest));
    printf("Sequence Number: %u\n", ntohs(tcp->seq));
    printf("Acknowledgment Number: %u\n", ntohs(tcp->ack_seq));
    printf("Data Offset: %d\n", tcp->doff * 4);
    printf("Reserved: 0x%04x\n", tcp->res1);
    printf("Flags:\n");
    printf(" URG: %d\n", (tcp->urg & 0x02) >> 1);
    printf(" ACK: %d\n", (tcp->ack & 0x10) >> 4);
    printf(" PSH: %d\n", (tcp->psh & 0x08) >> 3);
    printf(" RST: %d\n", (tcp->rst & 0x04) >> 2);
    printf(" SYN: %d\n", (tcp->syn & 0x02) >> 1);
    printf(" FIN: %d\n", tcp->fin & 0x01);
    printf("Window Size: %d\n", ntohs(tcp->window));
    printf("Checksum: 0x%04x\n", ntohs(tcp->check));
    printf("Urgent Pointer: %d\n", ntohs(tcp->urg_ptr));
}

// �в�ģ��:UDP ������غ���ʾ��
void print_udp(const struct udphdr *udp) {
    printf("UDP Header:\n");
    printf("Source Port: %d\n", ntohs(udp->source));
    printf("Destination Port: %d\n", ntohs(udp->dest));
    printf("Length: %d\n", ntohs(udp->len));
    printf("Checksum: 0x%04x\n", ntohs(udp->check));
}

// �в�ģ��:ICMP ������غ���ʾ��
void print_icmp(const struct icmphdr *icmp) {
    printf("ICMP Header:\n");
    printf("Type: %d\n", icmp->type);
    printf("Code: %d\n", icmp->code);
    printf("Checksum: 0x%04x\n", ntohs(icmp->checksum));

    // ��� ICMP �����Ƿ�Ϊ ECHOREPLY
    if (icmp->type == ICMP_ECHOREPLY) {
        // ���� icmp �ṹ��Ķ�����ȷ���� identifier ��Ա������Ӧ���� id ��Ա��
        printf("Identifier: %d\n", ntohs(icmp->un.echo.id));
        printf("Sequence Number: %d\n", ntohs(icmp->un.echo.sequence));
    }
}

// �������ݰ����ݷ�����������Ӧ��Э�鴦����(���ķ�������)
void analyze_packet(const void *packet, PacketStats *stats) {
    const struct ether_header *eth = (const struct ether_header *)packet;

    // �ж� MAC ��ַ���Ͳ�����
    if (memcmp(eth->ether_dhost, "\xff\xff\xff\xff\xff\xff", 6) == 0) {
        stats->macBroadcastCount++;
    } else if ((eth->ether_dhost[0] & 1) == 1) {
        // ������ж��鲥��ַ�������ֶ̺ͳ����ɸ���ʵ������ϸ����
        stats->macShortCount++;
    } else {
        stats->macLongCount++;
    }

    stats->macByteCount += sizeof(struct ether_header);
    stats->macPacketCount++;

    switch (ntohs(eth->ether_type)) {
        case ETHERTYPE_IP: {
            const struct iphdr *ip = (const struct iphdr *)(packet + sizeof(struct ether_header));

            // �ж� IP �㲥��ַ������
            if (ip->daddr == 0xffffffff) {
                stats->ipBroadcastCount++;
            }

            stats->ipByteCount += ntohs(ip->tot_len);
            stats->ipPacketCount++;

            print_ip(ip);

            switch (ip->protocol) {
                case IPPROTO_TCP: {
                    const struct tcphdr *tcp = (const struct tcphdr *)(packet + sizeof(struct ether_header) + (ip->ihl * 4));
                    print_tcp(tcp);
                    break;
                }
                case IPPROTO_UDP: {
                    const struct udphdr *udp = (const struct udphdr *)(packet + sizeof(struct ether_header) + (ip->ihl * 4));
                    print_udp(udp);
                    stats->udpPacketCount++;
                    break;
                }
                case IPPROTO_ICMP: {
                    const struct icmphdr *icmp = (const struct icmphdr *)(packet + sizeof(struct ether_header) + (ip->ihl * 4));
                    print_icmp(icmp);
                    stats->icmpPacketCount++;
                    // ��һ���ж� ICMP ���Ͳ�����
                    if (icmp->type == ICMP_REDIRECT) {
                        stats->icmpRedirectCount++;
                    } else if (icmp->type == ICMP_DEST_UNREACH) {
                        stats->icmpDestinationCount++;
                    }
                    break;
                }
                default:
                    printf("Unsupported IP protocol: %d\n", ip->protocol);
                    break;
            }
            break;
        }
        case ETHERTYPE_ARP:
            // ���������Ӷ� ARP Э�����ϸ�����߼�,Ŀǰ�򵥴�ӡ��ʾ
            printf("ARP packet received (not fully processed here)\n");
            break;
        default:
            printf("Unsupported Ethernet type: 0x%04x\n", ntohs(eth->ether_type));
            break;
    }
}

// ��ʼ�����ݰ�ͳ�ƽṹ��
void init_packet_stats(PacketStats *stats) {
    time_t current_time = time(NULL);
    struct tm *time_info = localtime(&current_time);
    strftime(stats->startTime, sizeof(stats->startTime), "%H:%M:%S %b %d %Y", time_info);

    stats->macBroadcastCount = 0;
    stats->macShortCount = 0;
    stats->macLongCount = 0;  // �������
    stats->macByteCount = 0;
    stats->macPacketCount = 0;
    stats->bitPerSecond = 0;
    stats->macByteSpeed = 0;
    stats->macPacketSpeed = 0;
    stats->ipBroadcastCount = 0;
    stats->ipByteCount = 0;
    stats->ipPacketCount = 0;
    stats->udpPacketCount = 0;
    stats->icmpPacketCount = 0;
    stats->icmpRedirectCount = 0;
    stats->icmpDestinationCount = 0;
}

// ��ȡ��ǰʱ�䲢���µ�ͳ�ƽṹ��Ľ���ʱ���ֶ�
void update_end_time(PacketStats *stats) {
    time_t current_time = time(NULL);
    struct tm *time_info = localtime(&current_time);
    strftime(stats->endTime, sizeof(stats->endTime), "%H:%M:%S %b %d %Y", time_info);
}
// ����Ƿ��а������µĺ���
int kbhit() {
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if (ch!= EOF) {
        ungetc(ch, stdin);
        return 1;
    }

    return 0;
}
// ���ݰ�Э��ͳ��ģ��,ͳ�Ʋ�ͬЭ������ݰ�����(������ϸҪ��ʵ��)
void protocol_stats(PacketStats *stats) {
    time_t startTime = time(NULL);
    double totalElapsedTime = 0;
    time_t lastTime = startTime;

    printf("Protocol Statistics:\n");
    printf("StartTime: %s\n", stats->startTime);

    // ѭ�������ۼ�ʱ����ͳ����Ϣ
    while (1) {
        time_t currentTime = time(NULL);
        double elapsedTime = difftime(currentTime, lastTime);
        totalElapsedTime += elapsedTime;
        lastTime = currentTime;

        // ����Ƿ��а������£������� 'e' ���˳�ѭ��
        if (kbhit()) {
            char ch = getchar();
            if (ch == 'e') {
                break;
            }
        }
    }

    // �����ۼ�ʱ����������
    if (totalElapsedTime > 0) {
        stats->bitPerSecond = (stats->macByteCount * 8) / totalElapsedTime;
        stats->macByteSpeed = stats->macByteCount / totalElapsedTime;
        stats->macPacketSpeed = stats->macPacketCount / totalElapsedTime;
    }

    update_end_time(stats);
    printf("EndTime: %s\n", stats->endTime);
    printf("MAC Broadcast: %d\n", stats->macBroadcastCount);
    printf("MAC Short: %d\n", stats->macShortCount);
    printf("MAC Long: %d\n", stats->macLongCount);
    printf("MAC Byte: %d\n", stats->macByteCount);
    printf("MAC Packet: %d\n", stats->macPacketCount);
    if (totalElapsedTime > 0) {
        printf("Bit/S: %d\n", stats->bitPerSecond);
        printf("MAC ByteSpeed: %d\n", stats->macByteSpeed);
        printf("MAC PacketSpeed: %d\n", stats->macPacketSpeed);
    } else {
        printf("Bit/S: N/A (Insufficient time elapsed)\n");
        printf("MAC ByteSpeed: N/A (Insufficient time elapsed)\n");
        printf("MAC PacketSpeed: N/A (Insufficient time elapsed)\n");
    }
    printf("IP Broadcast: %d\n", stats->ipBroadcastCount);
    printf("IP Byte: %d\n", stats->ipByteCount);
    printf("IP Packet: %d\n", stats->ipPacketCount);
    printf("UDP Packet: %d\n", stats->udpPacketCount);
    printf("ICMP Packet: %d\n", stats->icmpPacketCount);
    printf("ICMP Redirect: %d\n", stats->icmpRedirectCount);
    printf("ICMP Destination: %d\n\n\n", stats->icmpDestinationCount);
}




// ������,ģ���������ݰ���ؼ�ͳ������(ʾ��)
int main() {
    PacketStats stats;
    init_packet_stats(&stats);

    int sockfd = init_packet_capture("ens33", 65536);
    if (sockfd == -1) {
        return -1;
    }

    time_t lastPacketTime;  // �ڴ˴����� lastPacketTime

    while (1) {
        char packet_buffer[1500];
        ssize_t packet_size = recv(sockfd, packet_buffer, sizeof(packet_buffer), 0);
        if (packet_size == -1) {
            perror("Packet reception failed");
            continue;
        }
        analyze_packet(packet_buffer, &stats);
        // ����ͳ����Ϣ
        stats.macByteCount += packet_size;
        stats.macPacketCount++;
        if (ntohs(((struct ether_header *)packet_buffer)->ether_type) == ETHERTYPE_IP) {
            const struct iphdr *ip = (const struct iphdr *)(packet_buffer + sizeof(struct ether_header));
            stats.ipByteCount += ntohs(ip->tot_len);
            stats.ipPacketCount++;
            if (ip->protocol == IPPROTO_UDP) {
                stats.udpPacketCount++;
            } else if (ip->protocol == IPPROTO_ICMP) {
                stats.icmpPacketCount++;
            }
        }

        // ����ʱ��
        time_t current_time = time(NULL);
        lastPacketTime = current_time;

        if (kbhit()) {
            char ch = getchar();
            if (ch == 'e') {
                break;
            }
        }
    }

    protocol_stats(&stats);
    close(sockfd);
    return 0;
}
