void flash_led(uint8_t count) {
  do {
    T_PORT |= _BV(T_PIN);     // Set LED Pin High
    watchdogReset();          // Prevent reboot during delay
    _delay_ms(50);            // <--- Change for ON duration
    T_PORT &= ~(_BV(T_PIN));  // Set LED Pin Low
    _delay_ms(50);            // <--- Change for OFF duration
    watchdogReset();
  } while (--count);
}
// Change from 1 second to 500ms
watchdogConfig(WATCHDOG_500MS); 
