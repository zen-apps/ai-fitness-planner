docker exec 489ef15a6f15 bash -c "kill -9 \$(netstat -tulpn | grep ':7860' | awk '{print \$7}' | cut -d'/' -f1)"

ifconfig | grep "inet " | grep -v 127.0.0.1  

https://www.whatismyip.com/